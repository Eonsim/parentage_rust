#![feature(portable_simd)]
#![feature(let_chains)]
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use rust_htslib::bcf::record::GenotypeAllele::*;
use rust_htslib::bcf::record::{Genotype, GenotypeAllele};
use rust_htslib::bcf::{Read, Reader};
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::i32;
use std::io::BufRead;
use std::io::BufReader;
use std::io::{BufWriter, Write};
use std::simd::prelude::*;
use std::sync::mpsc;
use std::time::Instant;
const LANES: usize = 64;
const MINMARKERS: i32 = 90;
const MAXERRORS: f64 = 0.04;
//const MINMATCH: f64 = 0.99;
const POSMATCH: f64 = 0.97;
const DISCOVERY: i32 = 300;
const VER_MAX_ERR: i32 = 3;
const MIN_INF_MARKERS: i32 = 20;
const MAX_MARKERS: usize = 1907;

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
        fails = suminf - sumdifinf;
        end += inc;
        start += inc;
    }

    while start < psize && fails < tmperror {
        suminf += i32::from(child[start].abs() * parent[start].abs());
        sumdifinf += i32::from(child[start] * parent[start]);
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

/* Need, child, childgt, popmap, popgt, errors, ages,parent list*/
fn findparents(
    child: i32,
    childgt: &[i8],
    ped_parent: &i32,
    popmap: &HashMap<i32, usize>,
    pop_gts: &Vec<Vec<i8>>,
    allowed_errors: &i32,
    pos_parents: &BTreeSet<(i16, i32)>,
    ages: &HashMap<i32, i16>,
    inform_snp: &HashMap<usize, i32>,
) -> Vec<(i32, i32, i32, i32, f64)> {
    /* For possible parents First check pedpar if it matches return
        Otherwise if age is correct and parent has enough markers then parent match
    */
    let mut matches: Vec<(i32, i32, i32, i32, f64)> = Vec::with_capacity(1);
    let mut global: bool = true;
    if *ped_parent != 0
        && let Some(paridx) = popmap.get(ped_parent)
    {
        let pargt: &Vec<i8> = pop_gts.get(*paridx as usize).expect("couldn't unwrap");
        let my_pedpar: (i32, i32, i32, f64) = vec_pars(&childgt, &pargt, allowed_errors);
        let used_markers = my_pedpar.0 + my_pedpar.1;
        if my_pedpar.1 <= VER_MAX_ERR && used_markers >= MIN_INF_MARKERS {
            //MINMATCH {
            matches.push((
                *ped_parent,
                my_pedpar.0,
                my_pedpar.1,
                my_pedpar.2,
                my_pedpar.3,
            ));
            global = false;
            return matches;
        }
    }

    if global {
        for par in pos_parents {
            if let Some(paridx) = popmap.get(&par.1)
                && child != par.1
            {
                if let Some(infsnp) = inform_snp.get(paridx) {
                    if infsnp >= &DISCOVERY {
                        if let Some(cage) = ages.get(&child) {
                            if agecheck(cage, &par.0) {
                                let pargt: &Vec<i8> =
                                    pop_gts.get(*paridx as usize).expect("couldn't unwrap");
                                let pos_par: (i32, i32, i32, f64) =
                                    vec_pars(&childgt, &pargt, &allowed_errors);
                                let used_markers = pos_par.0 + pos_par.1;
                                if pos_par.3 >= POSMATCH && used_markers >= MIN_INF_MARKERS {
                                    matches
                                        .push((par.1, pos_par.0, pos_par.1, pos_par.2, pos_par.3));
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    matches
}

#[inline]
fn conv(gt: &str) -> (i8, i32) {
    match gt {
        "0/0" => return (-1i8, 1),
        "0/1" => return (0i8, 1),
        "1/1" => return (1i8, 1),
        //"0|0" => return (-1i8, 1),
        //"0|1" => return (0i8, 1),
        //"1|0" => return (0i8, 1),
        //"1|1" => return (1i8, 1),
        _ => return (0i8, 0),
    }
}

#[inline]
fn gtconv(gt: &[i32]) -> (i8, i32) {
    match gt {
        [2, 2] => return (-1i8, 1),
        [2, 4] => return (0i8, 1),
        [4, 4] => return (1i8, 1),
        //"0|0" => return (-1i8, 1),
        //"0|1" => return (0i8, 1),
        //"1|0" => return (0i8, 1),
        //"1|1" => return (1i8, 1),
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

    let vcf: &str = &args[1];
    let anmls_file: &String = &args[2];
    let ped_file: &String = &args[3];
    eprintln!("Loading VCF");
    let mut bcf = Reader::from_path(vcf).expect("Error opening file.");
    let _ = bcf.set_threads(6);
    let sample_count = usize::try_from(bcf.header().sample_count()).unwrap();
    let mut anml_lookup: HashMap<i32, usize> = HashMap::with_capacity(sample_count);
    let mut genotypes: Vec<Vec<i8>> = vec![Vec::with_capacity(MAX_MARKERS); sample_count];
    let mut inform: HashMap<usize, i32> = HashMap::with_capacity(sample_count);
    eprintln!("Indexing animals");
    let vcf_samples: Vec<&str> = bcf
        .header()
        .samples()
        .into_iter()
        .map(|x| std::str::from_utf8(x).expect("err"))
        .collect();
    for sample_idx in 0..sample_count {
        anml_lookup.insert(
            vcf_samples[sample_idx]
                .parse::<i32>()
                .expect("conversion error"),
            sample_idx,
        );
    }
    eprintln!("Starting GT load");

    for record in bcf.records().map(|r| r.expect("No record")) {
        //let b = rust_htslib::bcf::record::Buffer::new();
        //.enumerate() {
        //let record = records.expect("No record");
        let gts: Vec<(i8, i32)> = record
            .format(b"GT")
            .integer()
            .expect("GTs error")
            .to_vec()
            .iter()
            .map(|a| gtconv(a))
            .collect();
        //let gts = record.genotypes_shared_buffer(b).expect("Can't get GTs");
        //let mygts = rust_htslib::bcf::record::Genotype(gts);
        //for sample_index in 0..sample_count {
        for (mygt, sample_index) in gts.iter().zip(0..sample_count) {
            //let gt: (i8, i32) = conv(&gts.get(sample_index).to_string());
            //let gt: (i8, i32) = gtconv(gts[sample_index]);
            //let gt: (i8, i32) = gtconv(mygt);
            *inform.entry(sample_index).or_insert(0) += mygt.1;
            genotypes[sample_index].push(mygt.0);
        }
    }

    eprintln!("Read VCF");
    let tfile: File = File::open(&anmls_file).expect("Failed to read animal file");
    let mut anmls_list: Vec<i32> = Vec::with_capacity(anml_lookup.len());
    let areader: BufReader<File> = BufReader::new(tfile);
    eprintln!("Loading target anmls after {:?}", startt.elapsed());

    for line in areader.lines() {
        anmls_list.push(line.unwrap().parse::<i32>().unwrap());
    }

    let pfile = File::open(&ped_file).expect("Failed to read ped file");
    let mut sires_list: HashSet<i32> = HashSet::with_capacity(anml_lookup.len() / 30);
    let mut dams_list: HashSet<i32> = HashSet::with_capacity(anml_lookup.len());
    let mut ages: HashMap<i32, i16> = HashMap::with_capacity(anml_lookup.len());
    /* Sire, Dam, Sex, Birth*/
    let mut ped: HashMap<i32, (i32, i32, i8, i16)> = HashMap::with_capacity(anml_lookup.len());
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

    let fout = File::create("parentage_rust.csv").expect("Couldn't create file");
    let mut owrite = BufWriter::new(fout);
    let header = "Animal_Key,Sire_Verification_Code,Dam_Verification_Code,Number_Sire_Matches,Number_Dam_Matches,Sire_Match_1,Sire_Match_1_Number_Informative_SNP,Sire_Match_1_Pass_Rate,Dam_Match_1,Dam_Match_1_Number_Informative_SNP,Dam_Match_1_Pass_Rate,Sire_Match_2,Sire_Match_2_Number_Informative_SNP,Sire_Match_2_Pass_Rate,Dam_Match_2,Dam_Match_2_Number_Informative_SNP,Dam_Match_2_Pass_Rate\n";
    write!(owrite, "{}", header).expect("Can't write header");

    for an in anmls_list {
        if let Some(fam) = ped.get(&an) {
            let mut my_sires: Vec<(i32, i32, i32, i32, f64)> = vec![];
            let mut my_dams: Vec<(i32, i32, i32, i32, f64)> = vec![];
            if let Some(sires) = results.get(&an) {
                my_sires = sires.clone();
                my_sires.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            }
            if let Some(dams) = resultd.get(&an) {
                my_dams = dams.clone();
                my_dams.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            }
            let ped_sire = fam.0;
            let ped_dam = fam.1;
            let savail = anml_lookup.contains_key(&ped_sire);
            let davail = anml_lookup.contains_key(&ped_dam);
            write!(
                owrite,
                "{},{},{},{},{}",
                an,
                savail as i32,
                davail as i32,
                my_sires.len(),
                my_dams.len()
            )
            .expect("Can't write to file");

            for i in 0..2 {
                if let Some(smatch) = my_sires.get(i) {
                    write!(owrite, ",{},{},{}", smatch.0, smatch.1 + smatch.2, smatch.4)
                        .expect("Can't write to file");
                } else {
                    write!(owrite, ",0,0,0").expect("Can't write to file");
                }
                if let Some(dmatch) = my_dams.get(i) {
                    write!(owrite, ",{},{},{}", dmatch.0, dmatch.1 + dmatch.2, dmatch.4)
                        .expect("Can't write to file");
                } else {
                    write!(owrite, ",0,0,0").expect("Can't write to file");
                }
            }
            write!(owrite, "\n").expect("Can't write to file");
        }
    }

    owrite.flush().expect("Couldn't save file to disk");
    println!("Write finished");
}
