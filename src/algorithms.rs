use std::simd::prelude::*;
const LANES: usize = 64;
const RR: i8 = -1;
const RA: i8 = 0;
const AA: i8 = 1;
//const MS: i8 = 0;

#[inline(always)]
pub fn vec_pars(child: &[i8], parent: &[i8], max_err: &i32) -> (i32, i32, i32, f64) {
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
        suminf,
        f64::from(goodm) / f64::from(goodm + fails),
    )
}

pub fn trio_test_log(sirep: &[i8], damp: &[i8], childp: &[i8], maxfails: &i32) -> (bool, i32) {
    let mut fails = 0;
    let mut pass = 0;
    let mut valid_trio = true;
    for i in 0..childp.len() {
        let sgt = sirep[i];
        let dgt = damp[i];
        let cgt = childp[i];

        if cgt == RA && ((sgt == RR && dgt == AA) || (sgt == AA && dgt == RR)) {
            pass += 1;
        } else {
            if cgt == RA && ((sgt == RR && dgt == RR) || (sgt == AA && dgt == AA)) {
                fails += 1;
            } else {
                if (cgt == RR && sgt == RR && dgt == RR) || (cgt == AA && sgt == AA && dgt == AA) {
                    pass += 1
                }
            }
        }

        if fails > *maxfails {
            valid_trio = false;
            break;
        }
    }

    (valid_trio, fails)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> ([i8;66],[i8;66],[i8;66]){
        let sire : [i8; 66] =  [-1,-1,0,0,1,1,-1,-1,-1,0,1,0,0,0,0,0,-1,-1,0,0,1,1,-1,-1,0,0,1,1,0,0,0,-1,-1,-1,0,0,1,1,-1,-1,0,0,1,0,0,0,0,0,-1,-1,0,0,1,1,-1,-1,0,0,1,1,0,0,0,-1,0,0];
        let dam : [i8; 66] =   [1,-1,0,1,1,1,-1,-1,1,0,1,1,0,0,0,0,-1,-1,0,0,1,1,-1,-1,0,0,1,1,0,0,0,-1,1,-1,0,1,1,1,-1,-1,0,0,1,1,0,0,0,0,-1,-1,0,0,1,1,-1,-1,0,0,1,1,0,0,0,-1,-1,1];
        let child : [i8; 66] = [0,-1,0,1,1,1,-1,-1,0,0,1,1,0,0,0,0,-1,-1,0,0,1,1,-1,-1,0,0,1,1,0,0,0,0,0,-1,0,1,1,1,-1,-1,0,0,1,1,0,0,0,0,-1,-1,0,0,1,1,-1,-1,0,0,1,1,0,0,0,0,-1,1];
        (sire, dam, child)
    }

    #[test]
    fn trio_test_valid() {
        let data: ([i8; 66], [i8; 66], [i8; 66]) = setup();
        let myresult = trio_test_log(&data.0, &data.1, &data.2, &3);
        assert_eq!(myresult, (true,2));
    }

    #[test]
    fn trio_test_invalid() {
        let data = setup();
        let myresult = trio_test_log(&data.1, &data.2, &data.0, &3);
        assert_eq!(myresult, (false,4));
    }

    #[test]
    fn test_invalid_parent(){
        let data: ([i8; 66], [i8; 66], [i8; 66]) = setup();
        let result: (i32, i32, i32, f64) = vec_pars(&data.0, &data.1, &3);
        assert!(result.3 < 0.99);
    }

    #[test]
    fn test_valid_parent(){
        let data: ([i8; 66], [i8; 66], [i8; 66]) = setup();
        let result: (i32, i32, i32, f64) = vec_pars(&data.2, &data.1, &3);
        assert!(result.3 >= 0.99);
    }

}
