// #include "io/data_writer.hpp"
// #include <sstream>
// #include <iomanip>

// void DataWriter::writeSnapshot(const std::vector<double>& data, int timestep) {
//     try {
//         std::string filename = getFilename(timestep);
//         H5::H5File file(filename, H5F_ACC_TRUNC);

//         // Create dataspace and dataset
//         hsize_t dims[1] = {data.size()};
//         H5::DataSpace dataspace(1, dims);
//         H5::DataSet dataset = file.createDataSet("state", H5::PredType::NATIVE_DOUBLE, dataspace);
        
//         // Write data
//         dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
//     } catch (const H5::Exception& e) {
//         throw std::runtime_error("Failed to write HDF5 file");
//     }
// }

// std::string DataWriter::getFilename(int timestep) {
//     std::ostringstream oss;
//     oss << "output_" << std::setw(6) << std::setfill('0') << timestep << ".h5";
//     return oss.str();
// } 