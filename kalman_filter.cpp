#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

class KalmanFilter {
public:
    KalmanFilter(int state_dim, int measurement_dim)
        : state_dim_(state_dim), measurement_dim_(measurement_dim) {
        x_ = Eigen::VectorXd::Zero(state_dim);
        P_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
        F_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
        Q_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
        H_ = Eigen::MatrixXd::Identity(measurement_dim, state_dim);
        R_ = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
    }

    void predict() {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(const Eigen::VectorXd& z) {
        Eigen::VectorXd y = z - H_ * x_;
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
        
        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(state_dim_, state_dim_) - K * H_) * P_;
    }

    void set_state_transition(const Eigen::MatrixXd& F) { F_ = F; }
    void set_process_noise(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void set_measurement_matrix(const Eigen::MatrixXd& H) { H_ = H; }
    void set_measurement_noise(const Eigen::MatrixXd& R) { R_ = R; }
    
    Eigen::VectorXd get_state() const { return x_; }
    Eigen::MatrixXd get_covariance() const { return P_; }

private:
    int state_dim_;
    int measurement_dim_;
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_, F_, Q_, H_, R_;
};

namespace py = pybind11;

PYBIND11_MODULE(kalman_filter, m) {
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<int, int>())
        .def("predict", &KalmanFilter::predict)
        .def("update", &KalmanFilter::update)
        .def("set_state_transition", &KalmanFilter::set_state_transition)
        .def("set_process_noise", &KalmanFilter::set_process_noise)
        .def("set_measurement_matrix", &KalmanFilter::set_measurement_matrix)
        .def("set_measurement_noise", &KalmanFilter::set_measurement_noise)
        .def("get_state", &KalmanFilter::get_state)
        .def("get_covariance", &KalmanFilter::get_covariance);
}
