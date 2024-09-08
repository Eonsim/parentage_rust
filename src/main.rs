#![feature(portable_simd)]
#![feature(let_chains)]
use rand::seq::SliceRandom;
use rand::thread_rng;
use rust_htslib::bgzf::Reader;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::i32;
use std::io::BufRead;
use std::io::BufReader;
//use std::io::{BufWriter, Write};
use std::path::Path;
use std::simd::prelude::*;
use std::time::Instant;
const LANES: usize = 64;
const MINMARKERS: i32 = 90;
const MAXERRORS: f64 = 0.04;
const MINMATCH: f64 = 0.99;
const POSMATCH: f64 = 0.97;
const DISCOVERY: i32 = 300;

fn vec_pars(child: &[i8], parent: &[i8], max_err: &i32) -> (i32, i32, i32, f64) {
    let mut start: usize = 0;
    let mut end: usize = LANES;
    let mut suminf: i32 = 0;
    let mut sumdifinf: i32 = 0;
    let mut fails: i32 = 0;
    let tmperror: i32 = max_err * 2;
    let inc: usize = LANES;
    let psize: usize = child.len();
    let psi = psize as i32;

    while end < psize && fails < tmperror {
        let cvec: Simd<i8, LANES> = Simd::<i8, LANES>::from_slice(&child[start..end]);
        let pvec: Simd<i8, LANES> = Simd::<i8, LANES>::from_slice(&parent[start..end]);
        suminf += i32::from((cvec.abs() * pvec.abs()).reduce_sum());
        sumdifinf += i32::from((cvec * pvec).reduce_sum());
        //fails += suminf - sumdifinf; // should it be fails = as the sums are already acumulating
        fails = suminf - sumdifinf;
        end += inc;
        start += inc;
    }

    while start < psize && fails < tmperror {
        suminf += i32::from(child[start].abs() * parent[start].abs());
        sumdifinf += i32::from(child[start] * parent[start]);
        //fails += suminf - sumdifinf;
        fails = suminf - sumdifinf;
        start += 1;
    }

    fails = fails / 2;
    let uninf = psi - suminf; // as i32;
    let goodm = psi - (uninf + fails); // as i32;
    (
        goodm,
        fails,
        uninf,
        f64::from(goodm) / f64::from(goodm + fails),
    )
}

#[inline(always)]
fn agecheck(kid: &i16, par: &i16) -> bool {
    *kid - *par >= 2i16
}

fn findparents1(
    child: i32,
    childgt: &[i8],
    ped_parent: &i32,
    popmap: &HashMap<i32, i32>,
    pop_gts: &Vec<Vec<i8>>,
    allowed_errors: &i32,
    pos_parents: &BTreeSet<(i16, i32)>,
    ages: &HashMap<i32, i16>,
    inform_snp: &HashMap<i32, i32>,
) -> Vec<(i32, i32, i32, i32, f64)> {
    let mut matches: Vec<(i32, i32, i32, i32, f64)> = vec![];
    let mut global: bool = false;

    // Efficiently check for the pedigree parent
    if *ped_parent != 0i32 && let Some(paridx) = popmap.get(ped_parent) {
        let pargt: &Vec<i8> = pop_gts.get(*paridx as usize).expect("couldn't unwrap");
        let my_pedpar = vec_pars(&childgt, &pargt, allowed_errors);
        if my_pedpar.3 >= MINMATCH {
            matches.push((
                *ped_parent,
                my_pedpar.0,
                my_pedpar.1,
                my_pedpar.2,
                my_pedpar.3,
            ));
            return matches;
        } else {
            global = true;
        }
    } else {
        global = true;
    }

    // Optimize the loop over possible parents
    if global {
        // Use an iterator to avoid unnecessary allocations
        let mut pos_parents_iter = pos_parents.iter();
        while let Some(&par) = pos_parents_iter.next() {
            if let Some(paridx) = popmap.get(&par.1) {
                if let Some(cage) = ages.get(&child) {
                    if agecheck(cage, &par.0) {
                        if let Some(infsnp) = inform_snp.get(paridx) {
                            if infsnp >= &DISCOVERY {
                                let pargt: &Vec<i8> = pop_gts.get(*paridx as usize).expect("couldn't unwrap");
                                let pos_par = vec_pars(&childgt, &pargt, &allowed_errors);
                                if pos_par.3 >= POSMATCH {
                                    matches
                                        .push((par.1, pos_par.0, pos_par.1, pos_par.2, pos_par.3));
                                }
                            }
                        }
                    } else {
                        // Parents are sorted by age, so break early
                        break;
                    }
                }
            }
        }
    }

    matches
}

/* Need, child, childgt, popmap, popgt, errors, ages,parent list*/
fn findparents2(
    child: i32,
    childgt: &[i8],
    ped_parent: &i32,
    popmap: &HashMap<i32, i32>,
    pop_gts: &Vec<Vec<i8>>,
    allowed_errors: &i32,
    pos_parents: &BTreeSet<(i16, i32)>,
    ages: &HashMap<i32, i16>,
    inform_snp: &HashMap<i32, i32>,
) -> Vec<(i32, i32, i32, i32, f64)> {
    /* For possible parents First check pedpar if it matches return
        Otherwise if age is correct and parent has enough markers then parent match
    */
    let mut matches: Vec<(i32, i32, i32, i32, f64)> = vec![];
    let mut global: bool = false;
    if *ped_parent != 0i32
        && let Some(paridx) = popmap.get(ped_parent)
    {
        //let paridx: &i32 = popmap.get(ped_parent).unwrap();
        let pargt: &Vec<i8> = pop_gts.get(*paridx as usize).expect("couldn't unwrap");
        let my_pedpar: (i32, i32, i32, f64) = vec_pars(&childgt, &pargt, allowed_errors);
        if my_pedpar.3 >= MINMATCH {
            matches.push((
                *ped_parent,
                my_pedpar.0,
                my_pedpar.1,
                my_pedpar.2,
                my_pedpar.3,
            ));
            return matches;
        } else {
            global = true;
        }
    } else {
        global = true;
    }

    if global {
        for par in pos_parents {
            if let Some(paridx) = popmap.get(&par.1) {
                if let Some(cage) = ages.get(&child) {
                    if agecheck(cage, &par.0) {
                        if let Some(infsnp) = inform_snp.get(paridx) {
                            if infsnp >= &DISCOVERY {
                                let pargt: &Vec<i8> =
                                    pop_gts.get(*paridx as usize).expect("couldn't unwrap");
                                let pos_par: (i32, i32, i32, f64) =
                                    vec_pars(&childgt, &pargt, &allowed_errors);
                                if pos_par.3 >= POSMATCH {
                                    matches
                                        .push((par.1, pos_par.0, pos_par.1, pos_par.2, pos_par.3));
                                }
                            }
                        }
                    } else {
                        //Parents are sorted by age once one fails they all will
                        break;
                    }
                }
            }
        }
    }
    matches
}

fn findparents(
    child: i32,
    childgt: &[i8],
    ped_parent: &i32,
    popmap: &HashMap<i32, i32>,
    pop_gts: &[Vec<i8>],
    allowed_errors: &i32,
    pos_parents: &BTreeSet<(i16, i32)>,
    ages: &HashMap<i32, i16>,
    inform_snp: &HashMap<i32, i32>,
) -> Vec<(i32, i32, i32, i32, f64)> {
    let mut matches = Vec::new();
    let mut global = false;

    if *ped_parent != 0 {
        if let Some(&paridx) = popmap.get(ped_parent) {
            let pargt = &pop_gts[paridx as usize];
            let my_pedpar = vec_pars(childgt, pargt, allowed_errors);
            if my_pedpar.3 >= MINMATCH {
                matches.push((*ped_parent, my_pedpar.0, my_pedpar.1, my_pedpar.2, my_pedpar.3));
                return matches;
            } else {
                global = true;
            }
        } else {
            global = true;
        }
    } else {
        global = true;
    }

    if global {
        for &(age, parent) in pos_parents {
            if let Some(&paridx) = popmap.get(&parent) {
                if let Some(&cage) = ages.get(&child) {
                    if agecheck(&cage, &age) {
                        if let Some(&infsnp) = inform_snp.get(&paridx) {
                            if infsnp >= DISCOVERY {
                                let pargt = &pop_gts[paridx as usize];
                                let pos_par = vec_pars(childgt, pargt, allowed_errors);
                                if pos_par.3 >= POSMATCH {
                                    matches.push((parent, pos_par.0, pos_par.1, pos_par.2, pos_par.3));
                                }
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }

    matches
}


#[inline]
fn conv(gt: &str) -> (i8, i8) {
    match gt {
        "0/0" => return (-1i8, 1),
        "1/1" => return (1i8, 1),
        _ => return (0i8, 0),
    }
}

fn main() {
    let startt: Instant = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.contains(&String::from("-h")) {
        eprintln!(
            "Error, requires 3 files:\n myprog my.vcf.gz my.ids.txt my.ped threadsN\nExiting."
        );
        use std::process;
        process::exit(0);
    }

    let threads: &usize = if args.len() == 5 {
        &args[4].parse::<usize>().unwrap()
    } else {
        &8
    };

    let vcf: &String = &args[1];
    let anmls_file: &String = &args[2];
    let ped_file: &String = &args[3];
    let file: &Path = Path::new(&vcf);
    let reader: BufReader<Reader> = BufReader::new(Reader::from_path(file).unwrap());
    let mut first: bool = true;
    let mut anml_lookup: HashMap<i32, i32> = HashMap::new();
    let mut genotypes: Vec<Vec<i8>> = vec![];
    let mut count: i32 = 0;
    eprintln!("Loading VCF");
    /* Store number of informative markers */
    let mut inform: HashMap<i32, i32> = HashMap::new();

    for line in reader.lines() {
        //let dat: String = line.unwrap();
        let dat = line.expect("Failed to read line");
        if !dat.starts_with("##"){// contains("##") {
            if first {
                let tmpanmls: std::str::Split<'_, &str> = dat.split("\t");
                for an in tmpanmls.skip(9) {
                    anml_lookup
                        .entry(an.parse::<i32>().expect("Failed to parse"))
                        .or_insert(count);
                    inform.insert(count, 0);
                    genotypes.push(Vec::new());
                    //let blank: Vec<i8> = vec![];
                    //genotypes.push(blank);
                    count += 1;
                }
                first = false;
            } else {
                count = 0;
                let tmpgts: std::str::Split<'_, &str> = dat.split("\t");
                for tmpgt in tmpgts.skip(9) {
                    let gtconv: (i8, i8) = conv(tmpgt);
                    *inform.entry(count).or_insert(0) += i32::from(gtconv.1);
                    genotypes[count as usize].push(gtconv.0);
                    count += 1;
                }
            }
        }
    }
    let tfile: File = File::open(&anmls_file).expect("Failed to read animal file");
    let mut anmls_list: Vec<i32> = vec![];
    let areader: BufReader<File> = BufReader::new(tfile);
    eprintln!("Loading target anmls after {:?}", startt.elapsed());

    for line in areader.lines() {
        anmls_list.push(line.unwrap().parse::<i32>().unwrap());
    }

    let pfile = File::open(&ped_file).expect("Failed to read ped file");
    let mut sires_list: HashSet<i32> = HashSet::new();
    let mut dams_list: HashSet<i32> = HashSet::new();
    let mut ages: HashMap<i32, i16> = HashMap::new();
    /* Sire, Dam, Sex, Birth*/
    let mut ped: HashMap<i32, (i32, i32, i8, i16)> = HashMap::new();
    let preader: BufReader<File> = BufReader::new(pfile);
    eprintln!("Loading pedigree after {:?}", startt.elapsed());

    for line in preader.lines() {
        let tmp = line.unwrap();
        let dat: Vec<&str> = tmp.split_whitespace().collect();
        let child: i32 = dat[1].parse::<i32>().unwrap();
        let sire: i32 = dat[2].parse::<i32>().unwrap();
        let dam: i32 = dat[3].parse::<i32>().unwrap();
        let sex: i8 = dat[4].parse::<i8>().unwrap();
        let year: i16 = dat[5].parse::<i16>().unwrap();
        ped.insert(dat[1].parse::<i32>().unwrap(), (sire, dam, sex, year));
        if anml_lookup.contains_key(&sire) {
            sires_list.insert(sire);
        }
        if anml_lookup.contains_key(&sire) {
            dams_list.insert(dam);
        }
        if anml_lookup.contains_key(&child) {
            ages.insert(child, year);
            if sex == 1i8 {
                sires_list.insert(child);
            } else {
                dams_list.insert(child);
            }
        }
    }

    let mut sorted_sires = BTreeSet::new();
    for s in sires_list {
        if let Some(sireinfo) = ped.get(&s) {
            sorted_sires.insert((sireinfo.3, s));
        }
    }

    let mut sorted_dams = BTreeSet::new();
    for d in dams_list {
        if let Some(daminfo) = ped.get(&d) {
            sorted_dams.insert((daminfo.3, d));
        }
    }

    use rayon::prelude::*;
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let (txd, rxd) = mpsc::channel();

    let jobsize = anmls_list.len();
    let chunk_size = (jobsize / threads).max(1);
    eprintln!("Starting algorithm at {:?}", startt.elapsed());
    let algo_time = Instant::now();
    let mut rng = thread_rng();
    anmls_list.shuffle(&mut rng);

    anmls_list.par_chunks(chunk_size).for_each(|chunk| {
        let tx: mpsc::Sender<(i32, Vec<(i32, i32, i32, i32, f64)>)> = tx.clone();
        let txd: mpsc::Sender<(i32, Vec<(i32, i32, i32, i32, f64)>)> = txd.clone();
        for ban in chunk {
            if let Some(bidx) = anml_lookup.get(ban) {
                let bchild_gt: &Vec<i8> = &genotypes[*bidx as usize];
                if let Some(inf_markers) = inform.get(bidx) {
                    let maxerr: i32 = (f64::from(*inf_markers) * MAXERRORS) as i32;
                    if let Some(fam) = ped.get(ban)
                        && *inf_markers >= MINMARKERS
                    {
                        let sire_res: Vec<(i32, i32, i32, i32, f64)> = findparents(
                            *ban,
                            &bchild_gt,
                            &fam.0,
                            &anml_lookup,
                            &genotypes,
                            &maxerr,
                            &sorted_sires,
                            &ages,
                            &inform,
                        );
                        let dam_res: Vec<(i32, i32, i32, i32, f64)> = findparents(
                            *ban,
                            &bchild_gt,
                            &fam.1,
                            &anml_lookup,
                            &genotypes,
                            &maxerr,
                            &sorted_dams,
                            &ages,
                            &inform,
                        );
                        if sire_res.len() > 0 {
                            tx.send((*ban, sire_res)).expect("Thread error");
                        }
                        if dam_res.len() > 0 {
                            txd.send((*ban, dam_res)).expect("Thread error");
                        }
                    }
                }
            }
        }
    });
    eprintln!(
        "Algo time {:?}, total time {:?} with {} threads",
        algo_time.elapsed(),
        startt.elapsed(),
        threads
    );
    drop(tx);
    drop(txd);

    let results: HashMap<_, _> = rx.into_iter().collect();
    let resultd: HashMap<_, _> = rxd.into_iter().collect();

    for an in anmls_list {
        let mut my_sires: &Vec<(i32, i32, i32, i32, f64)> = &vec![];
        let mut my_dams: &Vec<(i32, i32, i32, i32, f64)> = &vec![];
        if let Some(sires) = results.get(&an) {
            my_sires = sires;
        }
        if let Some(dams) = resultd.get(&an) {
            my_dams = dams;
        }
        println!("{} s{:?} d{:?}", an, my_sires, my_dams);
    }
}
