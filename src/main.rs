use clap::{Arg, Command, value_parser};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, RgbaImage};
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};
use std::path::Path;

fn main() {
    let matches = Command::new("Butterworth Filter CLI")
        .version("1.0")
        .author("OpenAI")
        .about("Applies a Butterworth filter to an image")
        .arg(
            Arg::new("input")
                .help("Input image file path")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("output")
                .help("Output image file path")
                .required(true)
                .index(2),
        )
        .arg(
            Arg::new("type")
                .short('t')
                .long("type")
                .help("Type of Butterworth filter (lowpass or highpass)")
                .required(true)
                .value_parser(["lowpass", "highpass"]),
        )
        .arg(
            Arg::new("cutoff")
                .short('c')
                .long("cutoff-frequency")
                .help("Cutoff frequency for the Butterworth filter")
                .value_parser(value_parser!(f32))
                .default_value("50"),
        )
        .arg(
            Arg::new("order")
                .short('n')
                .long("order")
                .help("Order of the Butterworth filter")
                .value_parser(value_parser!(u32))
                .default_value("2"),
        )
        .arg(
            Arg::new("color")
                .short('k')
                .long("keep-color")
                .help("Keep color channels instead of converting to grayscale"),
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();
    let filter_type = matches.get_one::<String>("type").unwrap();
    let cutoff_frequency = *matches.get_one::<f32>("cutoff").unwrap();
    let order = *matches.get_one::<u32>("order").unwrap();
    let keep_color = matches.contains_id("color");

    let img = image::open(&Path::new(input_path)).expect("Failed to open input image");

    if keep_color {
        let filtered_img = apply_butterworth_filter_color(&img, filter_type, cutoff_frequency, order);
        filtered_img.save(output_path).expect("Failed to save output image");
    } else {
        let grayscale_img = img.to_luma8();
        let filtered_img = apply_butterworth_filter(&grayscale_img, filter_type, cutoff_frequency, order);
        filtered_img.save(output_path).expect("Failed to save output image");
    }
}

fn apply_butterworth_filter(
    img: &GrayImage,
    filter_type: &str,
    cutoff_frequency: f32,
    order: u32,
) -> GrayImage {
    let (width, height) = img.dimensions();
    let fft_width = width.next_power_of_two();
    let fft_height = height.next_power_of_two();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward((fft_width * fft_height) as usize);

    // Create frequency domain representation with padding
    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); (fft_width * fft_height) as usize];
    for y in 0..height {
        for x in 0..width {
            buffer[(y * fft_width + x) as usize] = Complex::new(img.get_pixel(x, y)[0] as f32, 0.0);
        }
    }
    fft.process(&mut buffer);

    // Apply Butterworth filter
    let mut filtered_buffer = buffer.clone();
    for y in 0..fft_height {
        for x in 0..fft_width {
            let d = (((x as f32 - fft_width as f32 / 2.0).powi(2)
                + (y as f32 - fft_height as f32 / 2.0).powi(2))
                .sqrt())
                / cutoff_frequency;

            let h = match filter_type.into() {
                "lowpass" => 1.0 / (1.0 + d.powi(2 * order as i32)),
                "highpass" => 1.0 / (1.0 + (1.0 / d).powi(2 * order as i32)),
                _ => {
                    eprintln!("Error: Invalid filter type '{}'. Use 'lowpass' or 'highpass'.", filter_type);
                    std::process::exit(1);
                }
            };

            let index = (y * fft_width + x) as usize;
            filtered_buffer[index] = buffer[index] * h;
        }
    }

    // Inverse FFT
    let ifft = planner.plan_fft_inverse((fft_width * fft_height) as usize);
    ifft.process(&mut filtered_buffer);

    // Normalize and convert back to image
    let epsilon = 1e-10;
    let min = filtered_buffer.iter().map(|c| c.re).fold(f32::INFINITY, f32::min);
    let max = filtered_buffer.iter().map(|c| c.re).fold(f32::NEG_INFINITY, f32::max);
    let mut output_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let val = ((filtered_buffer[(y * fft_width + x) as usize].re - min) / ((max - min) + epsilon) * 255.0) as u8;
            output_img.put_pixel(x, y, Luma([val]));
        }
    }

    output_img
}

fn apply_butterworth_filter_color(
    img: &DynamicImage,
    filter_type: &str,
    cutoff_frequency: f32,
    order: u32,
) -> RgbaImage {
    let (width, height) = img.dimensions();
    let fft_width = width.next_power_of_two();
    let fft_height = height.next_power_of_two();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward((fft_width * fft_height) as usize);

    let mut output_img = RgbaImage::new(width, height);

    for channel in 0..3 {
        // Extract channel with padding
        let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); (fft_width * fft_height) as usize];
        for y in 0..height {
            for x in 0..width {
                buffer[(y * fft_width + x) as usize] = Complex::new(img.to_rgba8().get_pixel(x, y)[channel] as f32, 0.0);
            }
        }

        let mut filtered_buffer = buffer.clone();
        fft.process(&mut filtered_buffer);

        // Apply Butterworth filter
        for y in 0..fft_height {
            for x in 0..fft_width {
                let d = (((x as f32 - fft_width as f32 / 2.0).powi(2)
                    + (y as f32 - fft_height as f32 / 2.0).powi(2))
                    .sqrt())
                    / cutoff_frequency;

                let h = match filter_type.into() {
                    "lowpass" => 1.0 / (1.0 + d.powi(2 * order as i32)),
                    "highpass" => 1.0 / (1.0 + (1.0 / d).powi(2 * order as i32)),
                    _ => {
                        eprintln!("Error: Invalid filter type '{}'. Use 'lowpass' or 'highpass'.", filter_type);
                        std::process::exit(1);
                    }
                };

                let index = (y * fft_width + x) as usize;
                filtered_buffer[index] = buffer[index] * h;
            }
        }

        // Inverse FFT
        let ifft = planner.plan_fft_inverse((fft_width * fft_height) as usize);
        ifft.process(&mut filtered_buffer);

        // Normalize and convert back to image
        let epsilon = 1e-10;
        let min = filtered_buffer.iter().map(|c| c.re).fold(f32::INFINITY, f32::min);
        let max = filtered_buffer.iter().map(|c| c.re).fold(f32::NEG_INFINITY, f32::max);

        for y in 0..height {
            for x in 0..width {
                let val = ((filtered_buffer[(y * fft_width + x) as usize].re - min) / ((max - min) + epsilon) * 255.0) as u8;
                output_img.get_pixel_mut(x, y)[channel] = val;
            }
        }
    }

    output_img
}
