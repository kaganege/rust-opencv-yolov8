// Rust implementation of https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-CPP-Inference/main.cpp

use gui::GUI;
use interface::Interface;
use keycode::KeyCode;
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture};
use opencv::{core, imgproc, Result};
use std::env;

mod gui;
mod interface;
mod keycode;

const RUN_ON_GPU: bool = false;
const MODEL_PATH: &str = "assets/yolov8n.onnx";
const WIDTH: i32 = 640;
const HEIGHT: i32 = 640;

fn main() -> Result<()> {
  let file_path = env::args()
    .nth(1)
    .expect("Please supply a video or image file!");

  let dev_count = core::get_cuda_enabled_device_count()?;
  let cuda_available = dev_count > 0;
  let cuda_support = cuda_available && !MODEL_PATH.ends_with(".onnx");

  println!(
    "CUDA is {}{}",
    if cuda_available {
      "available"
    } else {
      "not available"
    },
    if !cuda_support && cuda_available {
      " but the ai model does not support CUDA"
    } else {
      ""
    }
  );

  if cuda_available {
    for dev_num in 0..dev_count {
      core::print_short_cuda_device_info(dev_num)?;
    }
  }

  println!("Loading window...");
  let window = GUI::new(&file_path);
  // Add OpenGL support for GPU rendering gui (does not work)
  // window
  //   .add_window_flag(gui::WINDOW_OPENGL)?
  //   .set_opengl(true)?;
  window.init()?;

  println!("Initializing the video...");
  let mut capture = VideoCapture::from_file(file_path.as_str(), videoio::CAP_FFMPEG)?;
  if !capture.is_opened()? {
    eprintln!("Video or image cannot initialized!");
    return Ok(());
  }
  let is_video = capture.get(videoio::CAP_PROP_POS_FRAMES)? > 1.0;

  let mut inf = Interface::new(
    MODEL_PATH,
    core::Size::new(HEIGHT, WIDTH),
    "classes.txt",
    if cuda_support { RUN_ON_GPU } else { false },
  )?;

  'main_loop: loop {
    if window.is_closed()? {
      break;
    }

    // let mut frame = GpuMat::default()?;
    let mut frame = Mat::default();
    capture.read(&mut frame)?;

    let finished = frame.empty();

    if !finished {
      let detections = inf.run_interface(&frame)?;
      for detection in detections
        .iter()
        .filter(|d| d.class_id == 0 && d.confidence > 100.0)
      // Filter only persons
      {
        println!("Detection: {detection:?}");

        imgproc::rectangle(
          &mut frame,
          detection.box_,
          detection.color,
          2,
          imgproc::LINE_8,
          0,
        )?;
      }

      window.show_image(&frame)?;
    } else if is_video {
      println!("End of the video");
    }

    // do - while
    loop {
      if window.is_closed()? {
        break;
      }

      let key_code = window.wait_key(1)?;

      // F
      if key_code == KeyCode(102) {
        window.set_fullscreen(!window.is_fullscreen()?)?;
      }

      // ESC
      if key_code == KeyCode(27) {
        break 'main_loop;
      }

      if !finished {
        break;
      }
    }
  }

  Ok(())
}
