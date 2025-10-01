
using Printf
using Glob
using DataFrames
using CSV
using Statistics

# Define input and output paths
input_folder = "C:/Users/ANSUMAN/Downloads/Sahu SRF student/20240607/7.5m_5s"
output_raw_folder = "C:/Users/ANSUMAN/Downloads/NARL/processed/raw"
output_converted_folder = "C:/Users/ANSUMAN/Downloads/NARL/processed/converted"

# Create output directories if they do not exist
mkpath(output_raw_folder)
mkpath(output_converted_folder)

# Get list of all files in the input folder
input_files = glob("*", input_folder)

for input_path in input_files
    file_name = basename(input_path)
    output_raw_path = joinpath(output_raw_folder, "$(file_name)_raw.txt")
    output_converted_path = joinpath(output_converted_folder, "$(file_name)_converted.txt")

    open(input_path, "r") do input_file
        header_lines = [readline(input_file) for _ in 1:3]

        raw_data_values = UInt32[]
        while !eof(input_file)
            raw_data_value = read(input_file, UInt32)
            push!(raw_data_values, raw_data_value)
        end

        open(output_raw_path, "w") do output_raw_file
            for line in header_lines
                write(output_raw_file, line * "\r\n")
            end
            for raw_data_value in raw_data_values
                write(output_raw_file, @sprintf("%08X\r\n", raw_data_value))
            end
        end

        println("Raw data saved to $output_raw_path")


        # Convert raw data to decimal values
        raw_data_dec_values = map(Int, raw_data_values)

        # Noise correction
        num_data_points = length(raw_data_dec_values)
        half_index = floor(Int, num_data_points / 2)
        noise_average = mean(raw_data_dec_values[1:half_index])
        corrected_data = raw_data_dec_values .- noise_average

        # Range correction
        bin_width = 30
        range_correction = zeros(num_data_points)
        for i in 1:div(num_data_points, bin_width)
            range_start = (i - 1) * bin_width + 1
            range_end = min(i * bin_width, num_data_points)
            range = floor((range_start + range_end) / 2)
            range_correction[range_start:range_end] .= (range^2) * noise_average
        end

        corrected_combined_data = corrected_data .- range_correction

        open(output_converted_path, "w") do output_converted_file
            for line in header_lines
                write(output_converted_file, line * "\r\n")
            end
            for value in corrected_combined_data
                write(output_converted_file, @sprintf("%08X\r\n", value))
            end
        end

        println("Converted data saved to $output_converted_path")
    end
end