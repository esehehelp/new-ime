use anyhow::{bail, Context, Result};
use libloading::{Library, Symbol};
use std::ffi::{CStr, CString};
use std::io::{self, Write};
use std::path::PathBuf;

#[cfg(windows)]
use windows_sys::Win32::System::Console::{SetConsoleCP, SetConsoleOutputCP};

type FnVoidStrStr = unsafe extern "C" fn(*const i8, *const i8);
type FnVoidStr = unsafe extern "C" fn(*const i8);
type FnVoid = unsafe extern "C" fn();
type FnStrVoid = unsafe extern "C" fn() -> *const i8;
type FnFreeStr = unsafe extern "C" fn(*const i8);

struct EngineApi {
    _library: Library,
    initialize: FnVoidStrStr,
    shutdown: FnVoid,
    append_text: FnVoidStr,
    clear_text: FnVoid,
    get_composed_text: FnStrVoid,
    set_context: FnVoidStr,
    free_string: FnFreeStr,
}

impl EngineApi {
    fn load(path: &str) -> Result<Self> {
        let library =
            unsafe { Library::new(path) }.with_context(|| format!("failed to load DLL: {path}"))?;
        let initialize = unsafe { load_symbol::<FnVoidStrStr>(&library, b"Initialize\0")? };
        let shutdown = unsafe { load_symbol::<FnVoid>(&library, b"Shutdown\0")? };
        let append_text = unsafe { load_symbol::<FnVoidStr>(&library, b"AppendText\0")? };
        let clear_text = unsafe { load_symbol::<FnVoid>(&library, b"ClearText\0")? };
        let get_composed_text =
            unsafe { load_symbol::<FnStrVoid>(&library, b"GetComposedText\0")? };
        let set_context = unsafe { load_symbol::<FnVoidStr>(&library, b"SetContext\0")? };
        let free_string = unsafe { load_symbol::<FnFreeStr>(&library, b"FreeString\0")? };

        Ok(Self {
            _library: library,
            initialize,
            shutdown,
            append_text,
            clear_text,
            get_composed_text,
            set_context,
            free_string,
        })
    }

    fn initialize(&self, model_dir: &str) -> Result<()> {
        let model_dir = CString::new(model_dir)?;
        unsafe {
            (self.initialize)(model_dir.as_ptr(), std::ptr::null());
        }
        Ok(())
    }

    fn convert(&self, input: &str, context: Option<&str>) -> Result<String> {
        if let Some(context) = context.filter(|s| !s.is_empty()) {
            let context = CString::new(context)?;
            unsafe {
                (self.set_context)(context.as_ptr());
            }
        }

        let input = CString::new(input)?;
        unsafe {
            (self.clear_text)();
            (self.append_text)(input.as_ptr());
        }

        let raw = unsafe { (self.get_composed_text)() };
        if raw.is_null() {
            bail!("GetComposedText returned null");
        }

        let owned = unsafe { CStr::from_ptr(raw) }
            .to_str()
            .context("GetComposedText returned invalid UTF-8")?
            .to_owned();
        unsafe {
            (self.free_string)(raw);
        }
        Ok(owned)
    }
}

impl Drop for EngineApi {
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
        }
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    let symbol: Symbol<'_, T> = library.get(name)?;
    Ok(*symbol)
}

fn parse_args() -> (String, String) {
    let mut dll_path = "new-ime-runtime.dll".to_string();
    let mut model_dir = default_model_dir().to_string_lossy().to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dll" => {
                if let Some(value) = args.next() {
                    dll_path = value;
                }
            }
            "--models" => {
                if let Some(value) = args.next() {
                    model_dir = value;
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {}
        }
    }

    (dll_path, model_dir)
}

fn default_model_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cwd.join("models")
}

fn print_usage() {
    println!("Usage: cargo run -p new-ime-interactive -- [--dll PATH] [--models PATH]");
}

fn main() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);
    }

    let (dll_path, model_dir) = parse_args();
    let engine = EngineApi::load(&dll_path)?;

    println!("Initializing model...");
    engine.initialize(&model_dir)?;
    println!("Ready!\n");

    println!("=== new-ime Interactive Demo ===");
    println!("Type hiragana and press Enter to convert.");
    println!("Type 'q' to quit.\n");

    let mut context = String::new();
    let stdin = io::stdin();

    loop {
        if !context.is_empty() {
            let tail: String = context
                .chars()
                .rev()
                .take(20)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            println!("[context: ...{tail}]");
        }

        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }

        while matches!(line.chars().last(), Some('\n' | '\r')) {
            line.pop();
        }

        if matches!(line.as_str(), "q" | "quit") {
            break;
        }
        if line.is_empty() {
            continue;
        }

        let t0 = std::time::Instant::now();
        let result = engine.convert(&line, Some(&context))?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  => {result}  ({ms:.0}ms)\n");
        context = result;
    }

    println!("Bye!");
    Ok(())
}
