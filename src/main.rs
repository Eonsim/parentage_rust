#![feature(portable_simd)]
#![feature(let_chains)]
mod io_gsbin;
use io_gsbin::*;
mod algorithms;
use algorithms::*;
mod find_parents;
use find_parents::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::{BufWriter, Read, Write};
use std::sync::mpsc;
use std::time::Instant;
const MINMARKERS: i32 = 90;
const DISCOVERY_ACC: f64 = 0.99;
const DISCOVERY: i32 = 300;
const VER_MAX_ERR: i32 = 3;
const MIN_INF_MARKERS: i32 = 20;
//const MAX_MARKERS: usize = 1907;
const TRIO_ERROR: f64 = 0.04;
const MIN_PAR_AGE: i16 = 2;

fn main() {
    eprintln!("Rust Parent Match\nChad S. Harland\n Copyright (c) 2024");
    let startt: Instant = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.contains(&String::from("-h")) {
        eprintln!(
            "Error, requires 3 files:\n myprog my.vcf.gz my.ped my.ids.txt threadsN <--config config.json> <--debug>\nExiting."
        );
        use std::process;
        process::exit(0);
    }

    let debug_mode = if args.contains(&"--debug".to_string()) {
        true
    } else {
        false
    };
    let threads: &usize = if args.len() >= 5 {
        &args[4].parse::<usize>().unwrap()
    } else {
        &8
    };

    let mut min_markers = MINMARKERS;
    let mut min_informative = MIN_INF_MARKERS;
    let mut max_veri_errors = VER_MAX_ERR;
    let mut discovery_acc = DISCOVERY_ACC;
    let mut min_discovery = DISCOVERY;
    let mut trio_acc = TRIO_ERROR;
    let mut min_par_age = MIN_PAR_AGE;

    if args.contains(&"--config".to_string()) {
        eprintln!("Reading json configuration");
        use serde_json::Value;
        let json_idx = &args.iter().position(|x| x == "--config").unwrap() + 1;
        let mut jfile = File::open(&args[json_idx]).expect("can't Read config file");
        let mut config_json = String::new();
        jfile
            .read_to_string(&mut config_json)
            .expect("can't Read config file");
        let config: Value = serde_json::from_str(&config_json).expect("can't convert to JSON");
        min_markers = config["minimum_verification_snp"]
            .as_i64()
            .expect("Invalid JSON values") as i32;
        min_informative = config["minimum_informative_snp"]
            .as_i64()
            .expect("Invalid JSON values") as i32;
        min_discovery = config["minimum_discovery_snp"]
            .as_i64()
            .expect("Invalid JSON values") as i32;
        max_veri_errors = config["verification_maximum_snp_failures"]
            .as_i64()
            .expect("Invalid JSON values") as i32;
        discovery_acc = config["discovery_accuracy"]
            .as_f64()
            .expect("Invalid JSON values");
        trio_acc = config["trio_possible_accuracy"]
            .as_f64()
            .expect("Invalid JSON values");
        min_par_age = config["parent_child_age_difference"]
            .as_i64()
            .expect("Invalid JSON values") as i16;
    }
    let gtfile: &str = &args[1];
    let ped_file: &String = &args[2];
    let anmls_file: &String = &args[3];

    let myreader: fn(String) -> (HashMap<i32, usize>, Vec<Vec<i8>>, Vec<i32>) = read_gs;

    let (anml_lookup, genotypes, inform) = myreader(gtfile.to_string());

    let tfile: File = File::open(&anmls_file).expect("Failed to read animal file");
    let mut anmls_list: Vec<i32> = Vec::with_capacity(anml_lookup.len());
    let areader: BufReader<File> = BufReader::new(tfile);
    eprintln!("Loading target anmls after {:?}", startt.elapsed());

    for line in areader.lines() {
        let anml = line.unwrap().parse::<i32>().unwrap();
        if anml_lookup.contains_key(&anml) {
            anmls_list.push(anml);
        }
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
        let child: i32 = dat[1].parse::<i32>().expect("Couldn't parse Ped");
        let sire: i32 = dat[2].parse::<i32>().expect("Couldn't parse Ped");
        let dam: i32 = dat[3].parse::<i32>().expect("Couldn't parse Ped");
        let sex: i8 = dat[4].parse::<i8>().expect("Couldn't parse Ped");
        let year: i16 = dat[5].parse::<i16>().expect("Couldn't parse Ped");
        ped.insert(child, (sire, dam, sex, year));
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

    let mut sorted_sires: BTreeSet<(i16, i32, usize)> = BTreeSet::new();
    for s in sires_list {
        if let Some(sireinfo) = ped.get(&s) {
            if let Some(sire_idx) = anml_lookup.get(&s) {
                sorted_sires.insert((sireinfo.3, s, *sire_idx));
            }
        }
    }
    let sorted_sires: Vec<(i16, i32, usize)> = sorted_sires.into_iter().collect();

    let mut sorted_dams = BTreeSet::new();
    for d in dams_list {
        if let Some(daminfo) = ped.get(&d) {
            if let Some(dam_idx) = anml_lookup.get(&d) {
                sorted_dams.insert((daminfo.3, d, *dam_idx));
            }
        }
    }
    let sorted_dams: Vec<(i16, i32, usize)> = sorted_dams.into_iter().collect();

    let (tx, rx) = mpsc::channel();
    let (txd, rxd) = mpsc::channel();

    let jobsize = anmls_list.len();
    let chunk_size = (jobsize / threads).max(1);
    eprintln!("Starting algorithm at {:?}", startt.elapsed());
    let algo_time = Instant::now();
    let mut rng = thread_rng();
    anmls_list.shuffle(&mut rng);

    anmls_list.par_chunks(chunk_size).for_each(|chunk| {
        let tx: mpsc::Sender<(
            i32,
            (Vec<(i32, i32, i32, i32, f64)>, (i32, i32, i32, i32, f64)),
        )> = tx.clone();
        let txd: mpsc::Sender<(
            i32,
            (Vec<(i32, i32, i32, i32, f64)>, (i32, i32, i32, i32, f64)),
        )> = txd.clone();
        for ban in chunk {
            if let Some(bidx) = anml_lookup.get(ban) {
                let bchild_gt: &Vec<i8> = &genotypes[*bidx as usize];
                let inf_markers = inform[*bidx];
                let maxerr: i32 = (f64::from(inf_markers) * (1.0 - discovery_acc)) as i32;
                if let Some(fam) = ped.get(ban)
                    && inf_markers >= min_markers
                {
                    let sire_res: (Vec<(i32, i32, i32, i32, f64)>, (i32, i32, i32, i32, f64)) =
                        findparents(
                            *ban,
                            &bchild_gt,
                            &fam.0,
                            &anml_lookup,
                            &genotypes,
                            &maxerr,
                            &sorted_sires,
                            &ages,
                            &inform,
                            &min_informative,
                            &min_discovery,
                            &max_veri_errors,
                            &discovery_acc,
                            &min_par_age,
                        );
                    let dam_res: (Vec<(i32, i32, i32, i32, f64)>, (i32, i32, i32, i32, f64)) =
                        findparents(
                            *ban,
                            &bchild_gt,
                            &fam.1,
                            &anml_lookup,
                            &genotypes,
                            &maxerr,
                            &sorted_dams,
                            &ages,
                            &inform,
                            &min_informative,
                            &min_discovery,
                            &max_veri_errors,
                            &discovery_acc,
                            &min_par_age,
                        );
                    tx.send((*ban, sire_res)).expect("Thread error");
                    txd.send((*ban, dam_res)).expect("Thread error");
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
    write!(owrite, "Animal_Key,Sire_Verification_Code,Dam_Verification_Code,Number_Sire_Matches,Number_Dam_Matches").expect("Can't write header");
    //let header = ",Sire_Match_1,Sire_Match_1_Number_Informative_SNP,Sire_Match_1_Pass_Rate,Dam_Match_1,Dam_Match_1_Number_Informative_SNP,Dam_Match_1_Pass_Rate,Sire_Match_2,Sire_Match_2_Number_Informative_SNP,Sire_Match_2_Pass_Rate,Dam_Match_2,Dam_Match_2_Number_Informative_SNP,Dam_Match_2_Pass_Rate,Trio_Verification_Result,Trio_Verification_Sample_Swap_Check,Trio_Verification_Number_Informative_SNP,Trio_Verification_Pass_Rate";
    for i in 1..9 {
        write!(owrite, ",Sire_Match_{},Sire_Match_{}_Number_Informative_SNP,Sire_Match_{}_Pass_Rate,Dam_Match_{},Dam_Match_{}_Number_Informative_SNP,Dam_Match_{}_Pass_Rate",i,i,i,i,i,i).expect("Can't write header");
    }
    write!(owrite,",Trio_Verification_Result,Trio_Verification_Sample_Swap_Check,Trio_Verification_Number_Informative_SNP,Trio_Verification_Pass_Rate").expect("Can't write header");
    if debug_mode {
        write!(owrite, ",Ped_Sire,Ped_Dam,Trio").expect("Can't write header");
    }
    write!(owrite, "\n").expect("Can't write header");

    for an in anmls_list {
        if let Some(fam) = ped.get(&an) {
            let mut my_sires: Vec<(i32, i32, i32, i32, f64)> = vec![];
            let mut my_dams: Vec<(i32, i32, i32, i32, f64)> = vec![];
            let mut ped_sire_res: (i32, i32, i32, i32, f64) = (0, 0, 0, 0, 0.0);
            let mut ped_dam_res: (i32, i32, i32, i32, f64) = (0, 0, 0, 0, 0.0);
            let mut tswap = 0;
            let mut trio_check = -1;
            let mut trio: (i32, i32, f64) = (0, 0, 0.0);
            let childidx = &anml_lookup.get(&an).unwrap();
            let child_inform = inform[**childidx];
            if let Some(sires) = results.get(&an) {
                my_sires = sires.0.clone();
                my_sires.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                ped_sire_res = sires.1;
            }
            if let Some(dams) = resultd.get(&an) {
                my_dams = dams.0.clone();
                my_dams.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                ped_dam_res = dams.1;
            }
            if my_sires.len() > 0 && my_dams.len() > 0 {
                let childp: &Vec<i8> = &genotypes[**childidx];
                let child_error = (f64::from(child_inform) * (1.0 - trio_acc)) as i32;
                for s in &my_sires {
                    let sirep: &Vec<i8> = &genotypes[*anml_lookup.get(&s.0).unwrap()];
                    for d in &my_dams {
                        let trio_res = trio_test_log(
                            sirep,
                            &genotypes[*anml_lookup.get(&d.0).unwrap()],
                            childp,
                            &child_error,
                        );
                        //println!("{:?}", trio_res);
                        if trio_res.0 {
                            let pass_rate =
                                f64::from(child_inform - trio_res.1) / f64::from(child_inform);
                            trio = (s.0, d.0, pass_rate);
                            trio_check = 1;
                        } else {
                            let trio_res = trio_test_log(
                                sirep,
                                childp,
                                &genotypes[*anml_lookup.get(&d.0).unwrap()],
                                &child_error,
                            );
                            if trio_res.0 {
                                tswap = 1;
                                let pass_rate =
                                    f64::from(child_inform - trio_res.1) / f64::from(child_inform);
                                trio = (s.0, d.0, pass_rate);
                                trio_check = 1;
                            }
                        }
                    }
                }
            }

            let ped_sire: i32 = fam.0;
            let ped_dam: i32 = fam.1;
            let savail: bool = anml_lookup.contains_key(&ped_sire);
            let davail: bool = anml_lookup.contains_key(&ped_dam);
            write!(
                owrite,
                "{},{},{},{},{}",
                an,
                savail as i32,
                davail as i32,
                &my_sires.len(),
                &my_dams.len()
            )
            .expect("Can't write to file");

            for i in 0..8 {
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
            if trio_check == 1 {
                write!(
                    owrite,
                    ",{},{},{},{}",
                    trio_check, tswap, child_inform, trio.2
                )
                .expect("Can't write to file");
            } else {
                write!(owrite, ",NA,NA,0,0").expect("Can't write to file");
            }

            if debug_mode {
                let dsire: String = format!("{:?}", ped_sire_res)
                    .replace(",", "|")
                    .replace(" ", "");
                let ddam: String = format!("{:?}", ped_dam_res)
                    .replace(",", "|")
                    .replace(" ", "");
                let dtrio: String = format!("{:?}", trio).replace(",", "|").replace(" ", "");
                write!(owrite, ",{},{},{}", dsire, ddam, dtrio).expect("Can't write to file");
            }
            write!(owrite, "\n").expect("Can't write to file");
        }
    }

    owrite.flush().expect("Couldn't save file to disk");
    println!("Write finished");
}
