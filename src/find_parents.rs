use std::collections::HashMap;
use crate::algorithms::*;

#[inline(always)]
fn agecheck(kid: &i16, par: &i16, min_age: &i16) -> bool {
    *kid - *par >= *min_age
}

/* Need, child, childgt, popmap, popgt, errors, ages,parent list*/
pub fn findparents(
    child: i32,
    childgt: &[i8],
    ped_parent: &i32,
    popmap: &HashMap<i32, usize>,
    pop_gts: &Vec<Vec<i8>>,
    allowed_errors: &i32,
    pos_parents: &Vec<(i16, i32, usize)>,
    ages: &HashMap<i32, i16>,
    inform_snp: &Vec<i32>,
    min_informative: &i32,
    discover_snp: &i32,
    max_veri_error: &i32,
    discovery_acc: &f64,
    min_age: &i16,
) -> (Vec<(i32, i32, i32, i32, f64)>, (i32, i32, i32, i32, f64)) {
    /* For possible parents First check pedpar if it matches return
        Otherwise if age is correct and parent has enough markers then parent match
    */
    let mut used_markers: i32 = 0;
    let mut ped_match: (i32, i32, i32, i32, f64) = (0, 0, 0, 0, 0.0);
    let cage: &i16 = ages.get(&child).unwrap();
    let mut matches: Vec<(i32, i32, i32, i32, f64)> = Vec::with_capacity(2);
    let mut global: bool = true;
    if *ped_parent != 0
        && let Some(paridx) = popmap.get(ped_parent)
    {
        let pargt: &Vec<i8> = //pop_gts.get(*paridx as usize).expect("couldn't unwrap");
            &pop_gts[*paridx];
        let my_pedpar: (i32, i32, i32, f64) = vec_pars(&childgt, &pargt, allowed_errors);
        ped_match = (
            *ped_parent,
            my_pedpar.0,
            my_pedpar.1,
            my_pedpar.2,
            my_pedpar.3,
        );
        let used_markers = my_pedpar.0 + my_pedpar.1;
        if my_pedpar.1 <= *max_veri_error && used_markers >= *min_informative {
            matches.push(ped_match);
            global = false;
            return (matches, ped_match);
        }
    }

    if global {
        for par in pos_parents {
            if child != par.1 {
                if inform_snp[par.2] >= *discover_snp {
                    if agecheck(cage, &par.0, &min_age) {
                        let pos_par = vec_pars(&childgt, &pop_gts[par.2], &allowed_errors);
                        used_markers = pos_par.0 + pos_par.1;
                        if pos_par.3 >= *discovery_acc && used_markers >= *min_informative {
                            matches.push((par.1, pos_par.0, pos_par.1, pos_par.2, pos_par.3));
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }
    (matches, ped_match)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_age() {
        let myresult = agecheck(&2024, &2020, &2);
        assert!(myresult);
    }

    #[test]
    fn invalid_age() {
        let myresult = agecheck(&2019, &2020, &2);
        assert!(myresult == false);
    }


}