#![feature(portable_simd)]
use rust_htslib::bgzf::Reader;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::i32;
use std::io::BufRead;
use std::io::BufReader;
use rand::seq::SliceRandom;
use rand::thread_rng;
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

fn vec_pars(child: &[i8], parent: &[i8], max_err: &i32) -> (i32,i32,i32,f64) {
    let mut start: usize = 0;
    let mut end: usize = LANES;
    let mut suminf: i32 = 0;
    let mut sumdifinf: i32 = 0;
    let mut fails: i32 = 0;
    let tmperror: i32 = max_err * 2 ;
    let inc: usize = LANES;
    let psize: usize = child.len();

    while end < psize && fails < tmperror {
        let cvec: Simd<i8, LANES> = Simd::<i8, LANES>::from_slice(&child[start..end]);
        let pvec: Simd<i8, LANES> = Simd::<i8, LANES>::from_slice(&parent[start..end]);
        suminf += i32::from((cvec.abs() * pvec.abs()).reduce_sum());
        sumdifinf += i32::from((cvec * pvec).reduce_sum());
        fails += suminf - sumdifinf;
        end += inc;
        start += inc;
    }

    while start < psize && fails < tmperror{
        suminf += i32::from(child[start].abs() * parent[start].abs());
        sumdifinf += i32::from(child[start] * parent[start]);
        fails += suminf - sumdifinf;
        start += 1;
    }

    fails = fails / 2;
    let uninf = (psize as i32 - suminf) as i32;
    let goodm = (psize as i32 - (uninf + fails)) as i32;
    (goodm, fails, uninf , f64::from(goodm) /f64::from(goodm + fails))

}

#[inline(always)]
fn agecheck(kid: &i16, par: &i16) -> bool{
    *kid - 2i16 >= *par
}

/* Need, child, childgt, popmap, popgt, errors, ages,parent list*/
fn findparents(child:i32,
    childgt: &[i8],
    ped_parent: &i32,
    popmap: &HashMap<i32,i32>,
    pop_gts: &Vec<Vec<i8>>,
    allowed_errors: &i32,
    pos_parents: &HashSet<i32>,
    ages: &HashMap<i32,i16>,
    //ped: &HashMap<i32,(i32,i32,i8,i16)>,
    inform_snp: &HashMap<i32,i32>
    ) -> Vec<(i32,i32,i32,i32,f64)> {
    /* For possible parents First check pedpar if it matches return
        Otherwise if age is correct and parent has enough markers then parent match
    */
    let mut matches : Vec<(i32,i32,i32,i32,f64)> = vec![];
    let mut global : bool = false;
    if *ped_parent != 0i32 && popmap.contains_key(ped_parent){
        let paridx: &i32 = popmap.get(ped_parent).unwrap();
        let pargt : &Vec<i8> = pop_gts.get(*paridx as usize).expect("couldn't unwrap");
        let my_pedpar: (i32, i32, i32, f64) = vec_pars(&childgt, &pargt, allowed_errors);

        if my_pedpar.3 >= MINMATCH{
            matches.push((*ped_parent,my_pedpar.0,my_pedpar.1,my_pedpar.2,my_pedpar.3));
            return matches
        } else {
            global = true;
        }
    } else {
        global = true;
    }

    if global {
        for par in pos_parents{
            if ages.contains_key(par) && popmap.contains_key(par) && agecheck(ages.get(&child).unwrap(), ages.get(par).unwrap()) && inform_snp.get(popmap.get(par).unwrap()).unwrap() >= &DISCOVERY{
                let pargt : &Vec<i8>= pop_gts.get(*popmap.get(par).unwrap() as usize).expect("couldn't unwrap");
                let pos_par: (i32, i32, i32, f64) = vec_pars(&childgt, &pargt, &allowed_errors);
                if pos_par.3 >= POSMATCH {
                    matches.push((*par,pos_par.0,pos_par.1,pos_par.2,pos_par.3));
                }
            }
        }
    }
    matches
}

#[inline]
fn conv(gt: &str) -> (i8,i8) {
    match gt {
        "0/0" => return (-1i8,1),
        "1/1" => return (1i8,1),
        _ => return (0i8,0),
    }
}

fn main() {
    let startt: Instant = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.contains(&String::from("-h")){
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
        let dat: String = line.unwrap();
        if !dat.contains("##") {
            if first {
                let tmpanmls: std::str::Split<'_, &str> = dat.split("\t");
                for an in tmpanmls.skip(9) {
                    anml_lookup.insert(an.parse::<i32>().unwrap(), count);
                    inform.insert(count, 0);
                    let blank: Vec<i8> = vec![];
                    genotypes.push(blank);
                    count += 1;
                }
                first = false;
            } else {
                count = 0;
                let tmpgts: std::str::Split<'_, &str> = dat.split("\t");
                for tmpgt in tmpgts.skip(9) {
                    let gtconv: (i8, i8) = conv(&tmpgt);
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
    let mut ages: HashMap<i32,i16> = HashMap::new();
    /* Sire, Dam, Sex, Birth*/
    let mut ped: HashMap<i32, (i32,i32,i8,i16)> = HashMap::new();
    let preader: BufReader<File> = BufReader::new(pfile);
    eprintln!("Loading pedigree after {:?}", startt.elapsed());

    for line in preader.lines() {
        let tmp = line.unwrap();
        let dat : Vec<&str> = tmp.split_whitespace().collect();
        let child : i32 = dat[1].parse::<i32>().unwrap();
        let sire: i32 = dat[2].parse::<i32>().unwrap();
        let dam: i32 = dat[3].parse::<i32>().unwrap();
        let sex: i8 = dat[4].parse::<i8>().unwrap();
        let year: i16 = dat[5].parse::<i16>().unwrap();
        ped.insert(dat[1].parse::<i32>().unwrap(), (sire,dam,sex,year));
        if anml_lookup.contains_key(&sire) {sires_list.insert(sire);}
        if anml_lookup.contains_key(&sire) {dams_list.insert(dam);}
        if anml_lookup.contains_key(&child) {
            ages.insert(child,year);
            if sex == 1i8 {
                sires_list.insert(child);
            } else {
                dams_list.insert(child);
            }
        }
    }

    use rayon::prelude::*;
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();

    let jobsize = anmls_list.len();
    let chunk_size = (jobsize / threads).max(1);
    eprintln!("Starting algorithm at {:?}", startt.elapsed());
    let algo_time = Instant::now();
    let mut rng = thread_rng();
    anmls_list.shuffle(&mut rng);

    anmls_list.par_chunks(chunk_size).for_each(|chunk| {
        let tx: mpsc::Sender<(i32, Vec<(i32, i32, i32, i32, f64)>)> = tx.clone();
        for ban in chunk {
            //eprint!("an{}\n", ban);
            if anml_lookup.contains_key(ban) {
                let bidx: &i32 = anml_lookup.get(ban).unwrap();
                //eprint!("idx{}\n", bidx);
                let bchild_gt: &Vec<i8> = &genotypes[*bidx as usize];
                if inform.contains_key(bidx){
                    let inf_markers: &i32 = inform.get(bidx).unwrap();
                    //eprint!("inf{}\n", inf_markers);
                    let maxerr: i32 = (f64::from(*inf_markers) * MAXERRORS) as i32;
                    if ped.contains_key(ban) && *inf_markers >= MINMARKERS{
                        let fam = ped.get(ban).unwrap();
                        //let sire_res = findparents(*ban, &bchild_gt, &fam.0, &anml_lookup, &genotypes, &maxerr, &sires_list, &ages, &ped, &inform );
                        //let dam_res = findparents(*ban, &bchild_gt, &fam.1, &anml_lookup, &genotypes, &maxerr, &dams_list, &ages, &ped, &inform );
                        let sire_res: Vec<(i32, i32, i32, i32, f64)> = findparents(*ban, &bchild_gt, &fam.0, &anml_lookup, &genotypes, &maxerr, &sires_list, &ages, &inform );
                        let dam_res: Vec<(i32, i32, i32, i32, f64)> = findparents(*ban, &bchild_gt, &fam.1, &anml_lookup, &genotypes, &maxerr, &dams_list, &ages, &inform );
                        if sire_res.len() > 0{
                            tx.send((*ban,sire_res)).expect("Thread error");
                        }
                        if dam_res.len() > 0{
                            tx.send((*ban,dam_res)).expect("Thread error");
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
    let results : Vec<(i32,Vec<(i32,i32,i32,i32,f64)>)> = rx.iter().collect();

    for res in results{
        for pp in res.1{
            println!("{} {:?}",res.0,pp);
        }
    }

}
