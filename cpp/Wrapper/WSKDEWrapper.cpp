#include "WSKDEWrapper.h"
#include <iostream>

WSKDEWrapper::WSKDEWrapper(const py::array& H)
{
    py::gil_scoped_acquire acquire;
    py::module_ py_wskde_module = py::module_::import("wskde.wskde"); 
    py_wskde_wrapper = py_wskde_module.attr("WSKDE")(H);
}

void WSKDEWrapper::set_training_samples(const py::array& X_train, const py::array& Y_train)
{
    py::gil_scoped_acquire acquire;
    py_wskde_wrapper.attr("set_training_samples")(X_train, Y_train);
}

std::tuple<py::array, py::array> WSKDEWrapper::ws_kde(const py::array& x, const double& z)
{
    py::gil_scoped_acquire acquire;

    // Call Python ws_kde() and get the result (expect torch tensors or numpy arrays)
    py::object result = py_wskde_wrapper.attr("ws_kde")(x, z);

    // Convert result from torch::tensor to py::arrays
    py::object p0 = result.attr("__getitem__")(0);
    py::object p1 = result.attr("__getitem__")(1);
    py::array a0 = p0.attr("cpu")().attr("numpy")().cast<py::array>();
    py::array a1 = p1.attr("cpu")().attr("numpy")().cast<py::array>();

    return std::make_tuple(a0, a1);
}