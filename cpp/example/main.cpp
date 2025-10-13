#include <WSKDEWrapper.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <filesystem>
#include <random>
#include <cmath>

namespace py = pybind11;

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

py::array_t<double> vector_to_array(std::vector<double>& vec)
{
    return py::array(py::buffer_info(
        vec.data(),                                 // Allocate and copy data
        sizeof(double),                              // Size of one item
        py::format_descriptor<double>::format(),     // Data type format
        1,                                           // Number of dimensions
        { vec.size() },                              // Shape
        { sizeof(double) }                           // Strides
    ));
}


py::array_t<double> helper_create_bandwidth_matrix(const std::vector<double>& vec)
{
    std::vector<double> matrix(vec.size() * vec.size(), 0.0);
    for (std::size_t i = 0; i < vec.size(); ++i) {
        matrix[i * vec.size() + i] = vec[i];
    }
    
    const std::size_t n = vec.size();

    return py::array(py::buffer_info(
        matrix.data(),                                  // Allocate and copy data
        sizeof(double),                                 // Size of one item
        py::format_descriptor<double>::format(),        // Data type format
        2,                                              // Number of dimensions
        { vec.size(), vec.size() },                     // Shape
        { sizeof(double) * vec.size(), sizeof(double) } // Strides
    ));
}

py::array_t<double> helper_generate_random_array(std::size_t size, double min_val, double max_val)
{
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);

    // Allocate and fill vector with random doubles
    std::vector<double> result(size);
    for (std::size_t i = 0; i < size; ++i)
        result[i] = dis(gen);

    // Convert std::vector to py::array and return
    return vector_to_array(result);
}

py::array_t<double> helper_generate_random_binomial_array(const py::array& x_train)
{
    // Cast input to a typed numpy array and get a fast unchecked accessor
    py::array_t<double> arr = x_train.cast<py::array_t<double>>();
    auto buf = arr.unchecked<1>();
    std::size_t size = buf.shape(0);

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // Allocate and fill vector with random doubles based on sigmoid(x_train[i])
    std::vector<double> result(size);
    for (std::size_t i = 0; i < size; ++i)
        result[i] = (dis(gen) < sigmoid(buf(i))) ? 1.0 : 0.0;

    // Convert std::vector to py::array and return
    return vector_to_array(result);
}


py::array_t<double> helper_generate_uniform_array(std::size_t size, double min_val, double max_val)
{
    // Allocate and fill vector with random doubles
    std::vector<double> result(size);
    for (std::size_t i = 0; i < size; ++i)
        result[i] = min_val + (max_val - min_val) * i / (size - 1);

    // Convert std::vector to py::array and return
    return vector_to_array(result);
}

// Plot a py::array (1D) using matplotlib via pybind11
static void plot_wskde(const py::array& p_wskde, const py::array& sigma_wskde, const py::array& x_test, const py::array& x_train, const py::array& y_train)
{
    // Ensure matplotlib backend is non-interactive
    py::module_ mpl = py::module_::import("matplotlib");
    py::module_ np = py::module_::import("numpy");
    py::module_ plt = py::module_::import("matplotlib.pyplot");

    // Convert input arrays to typed numpy arrays and get fast unchecked accessors
    py::array_t<double> arr_x_test = x_test.cast<py::array_t<double>>();
    auto buf_x_test = arr_x_test.unchecked<1>();
    std::size_t size_x_test = buf_x_test.shape(0);
    py::array_t<double> arr_x_train = x_train.cast<py::array_t<double>>();
    auto buf_x_train = arr_x_train.unchecked<1>();
    std::size_t size_x_train = buf_x_train.shape(0);
    py::array_t<double> arr_y_train = y_train.cast<py::array_t<double>>();
    auto buf_y_train = arr_y_train.unchecked<1>();
    std::size_t size_y_train = buf_y_train.shape(0);

    // Sample Sigmoid function for reference
    std::vector<double> sigmoid_values(size_x_test);
    for (std::size_t i = 0; i < size_x_test; ++i)
        sigmoid_values[i] = sigmoid(buf_x_test(i));
    py::array y_sigmoid = vector_to_array(sigmoid_values);

    // Extract true and false values into separate arrays
    std::vector<double> x_true_success, x_true_failure;
    for (std::size_t i = 0; i < size_x_train; ++i)
    {
        if (buf_y_train(i) > 0.5)
            x_true_success.push_back(buf_x_train(i));
        else
            x_true_failure.push_back(buf_x_train(i));
    }
    py::array x_success = vector_to_array(x_true_success);
    py::array x_failure = vector_to_array(x_true_failure);
    std::vector<double> x_values(x_true_success.size(), -0.05);
    py::array y_success = vector_to_array(x_values);
    std::vector<double> y_values(x_true_failure.size(), -0.05);
    py::array y_failure = vector_to_array(y_values);

    // Plotting
    plt.attr("figure")();
    plt.attr("plot")(x_test, y_sigmoid, py::arg("linestyle") = "--", py::arg("color") = "black");
    plt.attr("plot")(x_success, y_success, py::arg("linestyle") = "None", py::arg("marker") = ".", py::arg("color") = "green");
    plt.attr("plot")(x_failure, y_failure, py::arg("linestyle") = "None", py::arg("marker") = "x", py::arg("color") = "red");
    plt.attr("plot")(x_test, p_wskde);
    plt.attr("fill_between")(x_test, p_wskde-sigma_wskde, p_wskde+sigma_wskde, py::arg("alpha") = 0.2);
    plt.attr("tight_layout")();
    plt.attr("show")();
}

int main() {
    // Setting the current path to align with the root folder
    std::filesystem::current_path("../../../");

    py::scoped_interpreter guard{};
    try {
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")("wskde");

        // Define the bandwidth matrix H (for this 1D case here only a scalar is needed)
        const double h = 0.5;
        py::array H = helper_create_bandwidth_matrix({h}); // 1x1 bandwidth matrix for 1D data

        // Define the training and test samples as numpy arrays
        const unsigned int number_of_samples = 100;
        py::array x_train = helper_generate_random_array(number_of_samples, -10.0, 10.0); // 100 random samples in [0, 100] -  this is the parameter vector
        py::array y_train = helper_generate_random_binomial_array(x_train);               // binomial samples based on sigmoid(X_train) - this is the outcome vector
        py::array x_test = helper_generate_uniform_array(101, -10.0, 10.0);               // 101 evenly spaced test samples in [0, 100] - this is where we want to evaluate the WSKDE

        // Example usage of WSKDEWrapper
        WSKDEWrapper* wskde = new WSKDEWrapper(H);     // Construct WSKDEWrapper with bandwidth matrix H
        wskde->set_training_samples(x_train, y_train); // Set training samples
        auto result = wskde->ws_kde(x_test, 1.96);     // Perform WSKDE on test samples with z=1.96

        // Extract results (now py::array)
        py::array p_wskde = std::get<0>(result);
        py::array sigma_wskde = std::get<1>(result);

        // Plot and save to files
        plot_wskde(p_wskde, sigma_wskde, x_test, x_train, y_train);

    } catch (const std::exception& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
