#![allow(unused_mut, unused_imports, dead_code)]
use anyhow::Result;
//use crypto::digest::Digest;
//use md5::{Context};
use sha2::{
    Sha256,
    Digest
};
use std::io::Read;

fn main() {
    let images = get_available_images().unwrap();
    println!("Get avaible images: {:?}\n{:?}", images.len(), images[0]);
}

#[derive(Debug)]
pub struct Image {
    pub filename: String,
    pub filepath: String,
    pub md5: String,
    //pub sha256: String,
}

/*pub fn get_available_images() -> Result<Vec<Image>> {
    let dir = r"C:\Users\11048\Pictures\images\pixabay.com\set1";
    let mut images = std::fs::read_dir(dir)?
        .filter_map(|file| {
            if let Ok(file) = file {
                if let Some(filename) = file.file_name().to_str() {
                    if filename.ends_with(".png") {
                        return Some(Image {
                            filename: filename.to_string(),
                        });
                    }
                }
            }
            None
        })
        .collect::<Vec<_>>();
    //models.append(&mut known_models);
    //models.sort_by(|a, b| b.custom.cmp(&a.custom));
    Ok(images)
}*/

/// Get local available images.
pub fn get_available_images() -> Result<Vec<Image>, std::io::Error> {
    let dir = r"C:\Users\11048\Pictures\images\pixabay.com\set1";
    let allowed_extensions = vec![
        ".png", 
        "jpg", 
        ".jpeg", 
        ".bmp"
    ];

    let mut images = std::fs::read_dir(dir)?
        .filter_map(|file| {
            if let Ok(file) = file {
                if let Some(filename) = file.file_name().to_str() {
                    if allowed_extensions.iter().any(|&ext| filename.ends_with(ext)) {
                        let filepath = format!("{dir}/{filename}");
                        return Some(Image {
                            filename: filename.to_string(),
                            filepath: filepath.clone(),
                            md5: calculate_md5(&std::path::PathBuf::from(filepath.clone())),
                            //sha256: calculate_sha256(&std::path::PathBuf::from(filepath.clone())),
                        });
                    }
                }
            }
            None
        })
        .collect::<Vec<_>>();
    
    Ok(images)
}

fn calculate_md5(file_path: &std::path::Path) -> String {
    let mut md5 = md5::Context::new();
    let mut buffer = [0; 4096];
    let mut file = std::fs::File::open(file_path).unwrap();

    loop {
        let bytes_read = file.read(&mut buffer).unwrap();
        if bytes_read == 0 {
            break;
        }
        md5.consume(&buffer[..bytes_read]);
    }

    format!("{:x}", md5.compute())
}

fn calculate_sha256(file_path: &std::path::Path) -> String {
    let mut sha256 = Sha256::new();
    let mut buffer = [0; 4096];
    let mut file = std::fs::File::open(file_path).unwrap();

    loop {
        let bytes_read = file.read(&mut buffer).unwrap();
        if bytes_read == 0 {
            break;
        }
        sha256.update(&buffer[..bytes_read]);
    }

	format!("{:x}", sha256.finalize())
}