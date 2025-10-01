using Printf, Statistics, Plots

function save_data_in_licel_format(input_folder, output_folder, threshold, initial_max_range, final_max_range, step)
    if !isdir(output_folder)
        mkdir(output_folder)
    end
    
    max_files = 2701
    
    files = readdir(input_folder)
    
    for current_max_range in initial_max_range:step:final_max_range
        combined_rtdi_data = zeros(UInt32, current_max_range, max_files)
        file_counter = 0
        
        for file in files
            if file_counter >= max_files
                break
            end
            
            input_file_path = joinpath(input_folder, file)
            
            if isfile(input_file_path)
                open(input_file_path, "r") do input_file
                    # Read header lines (if necessary)
                    header_lines = [readline(input_file) for _ in 1:3]

                    raw_data_values = UInt32[]
                    while !eof(input_file)
                        raw_data_value = read(input_file, UInt32)
                        push!(raw_data_values, raw_data_value)
                    end
                    
                    # Noise correction factor
                    noise_correcting_factor = mean(raw_data_values[current_max_range:2000])
                    noise_correcting_factor = round(Int, noise_correcting_factor)

                    # Apply noise and range corrections
                    for i in 5:min(current_max_range, length(raw_data_values))
                        raw_data_values[i] = max(raw_data_values[i] - noise_correcting_factor, 0)
                    end
                    
                    for i in 1:min(current_max_range, length(raw_data_values))
                        range = 15 + (i - 1) * 30
                        range_squared = UInt64(range) * UInt64(range)
                        corrected_value = UInt64(raw_data_values[i]) * range_squared
                        if corrected_value > UInt32(0xffffffff)
                            corrected_value = UInt32(0xffffffff)
                        end
                        combined_rtdi_data[i, file_counter+1] = UInt32(corrected_value)
                    end
                    
                    println("Processed file $file for max_range $current_max_range")
                    file_counter += 1
                end
            end
        end
        
        # Create and save heatmap
        plot_heatmap = heatmap(1:max_files, 1:current_max_range, combined_rtdi_data, c=:thermal, xlabel="File Index", ylabel="Range Bin", title="Combined RTDI Heatmap")
        
        # Generate filename with serial number
        figure_number = div((current_max_range - initial_max_range), step) + 50
        filename = joinpath(output_folder, "$(figure_number).png")
        savefig(plot_heatmap, filename)

        println("Combined RTDI heatmap saved as $filename")
    end
end

input_folder = "C:/Users/ANSUMAN/Downloads/Sahu SRF student/20240604_rain/7.5m_1s"
output_folder = "C:/Users/ANSUMAN/Downloads/NARL-Fig/Rainfall"
threshold = 0.5

initial_max_range = 500
final_max_range = 1000
step = 50

save_data_in_licel_format(input_folder, output_folder, threshold, initial_max_range, final_max_range, step)
