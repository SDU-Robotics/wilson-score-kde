#ifndef WSKDEWrapper_H
#define WSKDEWrapper_H

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <tuple>

namespace py = pybind11;

class WSKDEWrapper {
public:
    virtual ~WSKDEWrapper() = default;
    WSKDEWrapper(const py::array& H);

    void set_training_samples(const py::array& X_train, const py::array& Y_train);
    std::tuple<py::array, py::array> ws_kde(const py::array& x, const double& z = 1.96);

protected:
    py::object py_wskde_wrapper;
};

#endif // WSKDEWrapper_H
