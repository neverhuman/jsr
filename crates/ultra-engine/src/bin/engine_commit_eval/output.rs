//! Publish regular eval result files only after all manifest entries succeed.

use std::fs::{self, File, Permissions};
use std::io;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;

pub(super) fn open_output(path: &Path) -> io::Result<(File, Option<tempfile::TempPath>)> {
    let directory = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(directory)?;

    let permissions = match fs::symlink_metadata(path) {
        Ok(info) if !info.file_type().is_file() || info.nlink() > 1 => {
            // Retain direct writes for aliases and special paths such as /dev/null.
            return File::create(path).map(|file| (file, None));
        }
        Ok(_) => {
            // Check existing write access without erasing a previous result.
            let existing = File::options().write(true).open(path)?;
            Some(existing.metadata()?.permissions())
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => None,
        Err(err) => return Err(err),
    };

    let temporary = tempfile::Builder::new()
        .permissions(Permissions::from_mode(0o666))
        .tempfile_in(directory)?;
    if let Some(permissions) = permissions {
        temporary.as_file().set_permissions(permissions)?;
    }
    let (file, pending) = temporary.into_parts();
    Ok((file, Some(pending)))
}
