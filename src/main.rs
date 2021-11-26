use log::trace;
use ndarray::*;
use simplex_method::Table;

fn get_inverse_task(
    mut constr_coeff: ndarray::Array2<f64>,
    constr_val: ndarray::Array1<f64>,
    mut func_coeff: ndarray::Array1<f64>,
) -> Table {
    // AX >= B <=> -AX <= -B
    for i in constr_coeff.iter_mut() {
        *i *= -1f64;
    }
    for i in func_coeff.iter_mut() {
        *i *= -1f64;
    }
    Table::new(constr_coeff.reversed_axes(), func_coeff, constr_val, true)
}

fn main() {
    env_logger::init();
    // forward task
    let constr_coeff = array![[4f64, 1f64, 1f64], [1f64, 2f64, 0f64], [0f64, 0.5f64, 4f64]];
    let func_coeff = array![6f64, 6f64, 6f64];
    let constr_val = array![5f64, 3f64, 8f64];
    let mut table = Table::new(
        constr_coeff.clone(),
        constr_val.clone(),
        func_coeff.clone(),
        false,
    );
    trace!("Forward task:");
    let err_forward = table.optimise();
    let mut inverse_table = get_inverse_task(constr_coeff, constr_val, func_coeff);
    trace!("Inverse task:");
    let err_inverse = inverse_table.optimise();
    if err_forward.is_ok() && err_inverse.is_ok() {
        println!("Forward task:\n{}", table);
        println!("Inverse task:\n{}", inverse_table);
    } else if err_forward.is_err() && err_inverse.is_err() {
        println!(
            "Forward error: {:?}\tInverse error: {:?}",
            err_forward.unwrap_err(),
            err_inverse.unwrap_err()
        );
    } else {
        panic!("Stuff went horribly wrong");
    }
}
