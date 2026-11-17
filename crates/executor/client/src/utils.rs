/// Prints the marker that opens a profiled span in the zkVM's output. SP1 has two spellings
/// (`report` adds the span to its execution report); Pico's cycle tracker knows one.
#[cfg(target_os = "zkvm")]
pub fn span_start(name: &str, report: bool) {
    if cfg!(target_vendor = "pico") || !report {
        println!("cycle-tracker-start: {name}");
    } else {
        println!("cycle-tracker-report-start: {name}");
    }
}

/// Prints the marker that closes a span opened with [`span_start`].
#[cfg(target_os = "zkvm")]
pub fn span_end(name: &str, report: bool) {
    if cfg!(target_vendor = "pico") || !report {
        println!("cycle-tracker-end: {name}");
    } else {
        println!("cycle-tracker-report-end: {name}");
    }
}

/// Profile the given code block cycle count.
#[macro_export]
macro_rules! profile {
    ($name:expr, $block:block) => {{
        #[cfg(target_os = "zkvm")]
        {
            $crate::utils::span_start($name, false);
            let result = (|| $block)();
            $crate::utils::span_end($name, false);
            result
        }

        #[cfg(not(target_os = "zkvm"))]
        {
            $block
        }
    }};
}

/// Profile the given code block and add the cycle count to the execution report.
#[macro_export]
macro_rules! profile_report {
    ($name:expr, $block:block) => {{
        #[cfg(target_os = "zkvm")]
        {
            $crate::utils::span_start($name, true);
            let result = (|| $block)();
            $crate::utils::span_end($name, true);
            result
        }

        #[cfg(not(target_os = "zkvm"))]
        {
            $block
        }
    }};
}

pub use profile;
pub use profile_report;
