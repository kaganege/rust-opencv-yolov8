#![allow(dead_code)]

use super::keycode::KeyCode;
pub use highgui::{
  WindowFlags, WindowPropertyFlags, WINDOW_AUTOSIZE, WINDOW_FREERATIO, WINDOW_FULLSCREEN,
  WINDOW_GUI_EXPANDED, WINDOW_GUI_NORMAL, WINDOW_KEEPRATIO, WINDOW_NORMAL, WINDOW_OPENGL,
};
use opencv::core::{Size_, ToInputArray};
use opencv::{highgui, Result};

pub type WindowSize<T> = Size_<T>;

pub struct GUI {
  window_title: String,
  flags: i32,
}

impl Default for GUI {
  fn default() -> Self {
    GUI {
      window_title: "".to_string(),
      flags: WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED,
    }
  }
}

impl GUI {
  pub fn new(window_name: &str) -> Self {
    GUI {
      window_title: window_name.to_string(),
      ..Default::default()
    }
  }

  pub fn set_window_flags(&mut self, flags: i32) -> Result<&mut Self> {
    self.flags = flags;
    if !self.is_closed()? {
      self.destroy()?;
      self.init()?;
    }

    Ok(self)
  }

  pub fn add_window_flag(&mut self, flag: i32) -> Result<&mut Self> {
    self.set_window_flags(self.flags | flag)
  }

  fn get_window_property(&self, property: WindowPropertyFlags) -> Result<bool> {
    let value = highgui::get_window_property(&self.window_title, property as i32)?;

    Ok(value != 0.0)
  }

  fn set_window_property(&self, property: WindowPropertyFlags, value: bool) -> Result<&Self> {
    highgui::set_window_property(&self.window_title, property as i32, value as u8 as f64)?;

    Ok(self)
  }

  pub fn is_fullscreen(&self) -> Result<bool> {
    self.get_window_property(WindowPropertyFlags::WND_PROP_FULLSCREEN)
  }

  pub fn set_fullscreen(&self, enabled: bool) -> Result<&Self> {
    self.set_window_property(WindowPropertyFlags::WND_PROP_FULLSCREEN, enabled)
  }

  pub fn is_top_most(&self) -> Result<bool> {
    self.get_window_property(WindowPropertyFlags::WND_PROP_TOPMOST)
  }

  pub fn set_top_most(&self, enabled: bool) -> Result<&Self> {
    self.set_window_property(WindowPropertyFlags::WND_PROP_TOPMOST, enabled)
  }

  pub fn set_opengl(&self, enabled: bool) -> Result<&Self> {
    self.set_window_property(WindowPropertyFlags::WND_PROP_OPENGL, enabled)
  }

  pub fn set_vsync(&self, enabled: bool) -> Result<&Self> {
    self.set_window_property(WindowPropertyFlags::WND_PROP_VSYNC, enabled)
  }

  pub fn get_window_title(&self) -> String {
    self.window_title.clone()
  }

  pub fn set_window_title(&mut self, title: &str) -> Result<&mut Self> {
    highgui::set_window_title(&self.window_title, title)?;
    self.window_title = title.to_string();

    Ok(self)
  }

  pub fn is_closed(&self) -> Result<bool> {
    let is_visible = self.get_window_property(WindowPropertyFlags::WND_PROP_VISIBLE)?;

    Ok(!is_visible)
  }

  pub fn init(&self) -> Result<()> {
    if !self.is_closed()? {
      return Err(opencv::Error {
        code: 999,
        message: "That window is already initialized!".to_string(),
      });
    }

    highgui::named_window(&self.window_title, self.flags)
  }

  pub fn destroy(&self) -> Result<()> {
    if self.is_closed()? {
      return Ok(());
    }

    highgui::destroy_window(self.window_title.as_str())
  }

  pub fn is_resizable(&self) -> Result<bool> {
    let is_auto_size = self.get_window_property(WindowPropertyFlags::WND_PROP_AUTOSIZE)?;

    Ok(!is_auto_size)
  }

  pub fn resize(&self, (width, height): (i32, i32)) -> Result<()> {
    if !self.is_resizable()? {
      return Err(opencv::Error {
        code: 999,
        message: "That window cannot resizable!".to_string(),
      });
    }

    highgui::resize_window(&self.window_title, width, height)
  }

  pub fn update(&self) -> Result<()> {
    if self.is_closed()? {
      return Err(opencv::Error {
        code: 999,
        message: "A closed window cannot be updated!".to_string(),
      });
    }

    highgui::update_window(&self.window_title)
  }

  pub fn show_image(&self, mat: &impl ToInputArray) -> Result<()> {
    if self.is_closed()? {
      return Err(opencv::Error {
        code: 999,
        message: "A closed window cannot show an image!".to_string(),
      });
    }

    highgui::imshow(self.window_title.as_str(), mat)
  }

  pub fn create_trackbar(
    &self,
    trackbar_name: &str,
    value: Option<&mut i32>,
    count: i32,
    on_change: highgui::TrackbarCallback,
  ) -> Result<i32> {
    if self.is_closed()? {
      return Err(opencv::Error {
        code: 999,
        message: "Cannot crate a trackbar in closed window!".to_string(),
      });
    }

    highgui::create_trackbar(
      trackbar_name,
      self.window_title.as_str(),
      value,
      count,
      on_change,
    )
  }

  pub fn wait_key(&self, delay: i32) -> Result<KeyCode> {
    // if self.is_closed()? {
    //   return Ok(KeyCode(0));
    // }

    let key_code = highgui::wait_key(delay)?;

    Ok(KeyCode(key_code))
  }

  pub fn keep_alive(&self) -> Result<()> {
    loop {
      if self.is_closed()? {
        break;
      }

      self.update()?;
    }

    Ok(())
  }
}
