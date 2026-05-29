# Filter Calculation Steps

This document is synchronized with the step annotations in:

- `filters/kalman_filter.py`
- `filters/extended_kalman_filter.py`
- `filters/unscented_kalman_filter.py`
- `filters/particle_filter.py`
- `filters/invariant_kalman_filter.py`
- `filters/invariant_kalman_filter_15D.py`

The labels in code comments, for example `KF-P2` or `UKF-U4`, are the
sequence numbers for the equations below.

## Label Convention

Each code annotation uses this format:

```text
<FILTER>-<PHASE><STEP_NUMBER>
```

Examples:

- `KF-P2`: Kalman Filter, prediction step 2
- `EKF-U3`: Extended Kalman Filter, measurement update step 3
- `UKF-P4`: Unscented Kalman Filter, prediction step 4
- `PF-R1`: Particle Filter, resampling step 1
- `INEKF-U3`: Invariant EKF, measurement update step 3
- `INEKF15-U3`: 15D Invariant EKF, measurement update step 3

Phase meanings:

- `I`: initialization
- `P`: prediction
- `U`: measurement update
- `R`: resampling
- `S`: step orchestration

## Shared Constant-Velocity Model

KF, EKF, UKF, and PF use the same benchmark state:

$$
x_k =
\begin{bmatrix}
p_k \\
v_k
\end{bmatrix},
\quad
p_k, v_k \in \mathbb{R}^{d},
\quad
d \in \{2, 3\}.
$$

For the 3D benchmark, \(d=3\), so \(x_k \in \mathbb{R}^6\). The default
measurement is position:

$$
z_k = H_k x_k + r_k,
\quad
r_k \sim \mathcal{N}(0, R_k).
$$

The transition matrix and optional acceleration-control vector are:

$$
F_k =
\begin{bmatrix}
I_d & \Delta t I_d \\
0 & I_d
\end{bmatrix},
\quad
b_k =
\begin{bmatrix}
\frac{1}{2} a_k \Delta t^2 \\
a_k \Delta t
\end{bmatrix},
\quad
a_k = u_k - b_a + g.
$$

When no acceleration control is configured, \(b_k = 0\).

## Linear Kalman Filter

`KF-P1`

$$
F_k = F(\Delta t),
\quad
Q_k = \operatorname{diag}(q).
$$

`KF-P2`

$$
x_k^- = F_k x_{k-1} + b_k.
$$

`KF-P3`

$$
P_k^- = F_k P_{k-1} F_k^\top + Q_k.
$$

`KF-U1`

$$
z_k \in \mathbb{R}^{m}
$$

is validated against the configured measurement dimension \(m\).

`KF-U2`

$$
H_k = H,
\quad
R_k = \operatorname{diag}(r).
$$

`KF-U3`

$$
y_k = z_k - H_k x_k^-,
\quad
S_k = H_k P_k^- H_k^\top + R_k,
$$

$$
K_k = P_k^- H_k^\top S_k^{-1},
\quad
x_k = x_k^- + K_k y_k.
$$

The covariance uses the Joseph form:

$$
P_k =
(I - K_k H_k)P_k^-(I - K_k H_k)^\top
+ K_k R_k K_k^\top.
$$

`KF-S1` runs prediction, `KF-S2` runs measurement update, and `KF-S3`
returns the benchmark pose vector.

## Extended Kalman Filter

The EKF file keeps the EKF sequence explicit, but the comparable benchmark
model is the same constant-velocity model as the KF. Therefore, here
\(f(\cdot)\) and \(h(\cdot)\) are linear:

$$
f(x_{k-1}, u_k, \Delta t) = F_k x_{k-1} + b_k,
\quad
h(x_k^-) = H_k x_k^-.
$$

`EKF-P1`

$$
F_k =
\left.\frac{\partial f}{\partial x}\right|_{x_{k-1}},
\quad
Q_k = \operatorname{diag}(q).
$$

`EKF-P2`

$$
x_k^- = f(x_{k-1}, u_k, \Delta t).
$$

`EKF-P3`

$$
P_k^- = F_k P_{k-1} F_k^\top + Q_k.
$$

`EKF-U1`

$$
z_k \in \mathbb{R}^{m}
$$

is validated against the configured measurement dimension \(m\).

`EKF-U2`

$$
H_k =
\left.\frac{\partial h}{\partial x}\right|_{x_k^-},
\quad
R_k = \operatorname{diag}(r).
$$

`EKF-U3`

$$
y_k = z_k - h(x_k^-),
\quad
S_k = H_k P_k^- H_k^\top + R_k,
$$

$$
K_k = P_k^- H_k^\top S_k^{-1},
\quad
x_k = x_k^- + K_k y_k.
$$

The covariance uses the same Joseph form as `KF-U3`.

`EKF-S1` runs prediction, `EKF-S2` runs measurement update, and `EKF-S3`
returns the benchmark pose vector.

## Unscented Kalman Filter

Let \(n\) be the state dimension. The Merwe sigma-point parameters are:

$$
\lambda = \alpha^2(n + \kappa) - n.
$$

`UKF-P1`

$$
\chi_{0,k-1} = x_{k-1},
\quad
\chi_{i,k-1} = x_{k-1} \pm
\left[\sqrt{(n+\lambda)P_{k-1}}\right]_i.
$$

`UKF-P2`

$$
\chi_{i,k}^- = f(\chi_{i,k-1}, u_k, \Delta t).
$$

`UKF-P3`

$$
x_k^- = \sum_i W_i^m \chi_{i,k}^-,
$$

$$
P_k^- =
\sum_i W_i^c(\chi_{i,k}^- - x_k^-)(\chi_{i,k}^- - x_k^-)^\top
+ Q_k.
$$

`UKF-P4` regenerates sigma points around \(x_k^-\) and \(P_k^-\) for the
measurement update.

`UKF-U1`

$$
\zeta_{i,k} = h(\chi_{i,k}^-).
$$

`UKF-U2`

$$
\hat{z}_k = \sum_i W_i^m \zeta_{i,k},
$$

$$
S_k =
\sum_i W_i^c(\zeta_{i,k} - \hat{z}_k)(\zeta_{i,k} - \hat{z}_k)^\top
+ R_k.
$$

`UKF-U3`

$$
P_{xz,k} =
\sum_i W_i^c(\chi_{i,k}^- - x_k^-)(\zeta_{i,k} - \hat{z}_k)^\top,
\quad
K_k = P_{xz,k} S_k^{-1}.
$$

`UKF-U4`

$$
y_k = z_k - \hat{z}_k,
\quad
x_k = x_k^- + K_k y_k,
\quad
P_k = P_k^- - K_k S_k K_k^\top.
$$

`UKF-S1` runs prediction, `UKF-S2` runs measurement update, and `UKF-S3`
returns the benchmark pose vector.

## Particle Filter

The PF uses the same constant-velocity transition and position measurement
model, but represents the posterior with weighted samples:

$$
\{x_k^i, w_k^i\}_{i=1}^{N}.
$$

`PF-I1`

$$
x_0^i \sim \mathcal{N}(x_0, P_0).
$$

`PF-I2`

$$
w_0^i = \frac{1}{N}.
$$

`PF-P1`

$$
\bar{x}_k^i = F_k x_{k-1}^i + b_k.
$$

`PF-P2`

$$
x_k^i = \bar{x}_k^i + \epsilon_k^i,
\quad
\epsilon_k^i \sim \mathcal{N}(0, Q_k).
$$

`PF-U1`

$$
e_k^i = h(x_k^i) - z_k.
$$

`PF-U2`

$$
\tilde{w}_k^i =
w_{k-1}^i
\mathcal{N}(e_k^i; 0, R_k).
$$

`PF-U3`

$$
w_k^i =
\frac{\tilde{w}_k^i}{\sum_j \tilde{w}_k^j}.
$$

`PF-R1`

$$
N_{\mathrm{eff}} =
\frac{1}{\sum_i (w_k^i)^2}.
$$

If \(N_{\mathrm{eff}}\) is below the configured threshold, particles are
resampled and weights are reset to \(1/N\).

`PF-M1`

If measurement rejuvenation is enabled, position particles are refreshed around
the latest position measurement:

$$
p_k^i[\mathrm{indices}] =
z_k + \eta_k^i,
\quad
\eta_k^i \sim \mathcal{N}(0, R_k).
$$

When a previous position measurement is available, the matching velocity
components are refreshed around the finite-difference measurement velocity:

$$
\hat{v}_k =
\frac{z_k - z_{k-1}}{\Delta t_{\mathrm{meas}}},
\quad
v_k^i[\mathrm{indices}] =
\hat{v}_k + \nu_k^i.
$$

This keeps the constant-velocity particle state consistent after position
rejuvenation. Without this velocity refresh, position-only rejuvenation can
break the correlation between \(p\) and \(v\), causing drift between sparse
measurement updates.

`PF-S1` runs prediction, `PF-S2` runs likelihood update, `PF-S3` checks
resampling, `PF-S4` optionally applies measurement rejuvenation, and `PF-S5`
returns the weighted-mean benchmark pose.

## Invariant EKF

The default `filters/invariant_kalman_filter.py` InEKF is intentionally not
the same vector-space model as KF/EKF/UKF/PF.
Its nominal state is on \(SE_2(3)\):

$$
X =
\begin{bmatrix}
R & v & p \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix},
\quad
\xi =
\begin{bmatrix}
\phi \\
\rho_v \\
\rho_p
\end{bmatrix}
\in \mathbb{R}^{9}.
$$

`INEKF-P1`

Bias-correct the control input:

$$
u_k^c =
\begin{bmatrix}
u_{a,k} - b_a \\
u_{\omega,k} - b_g
\end{bmatrix},
$$

then propagate the nominal \(R_k^-\), \(v_k^-\), and \(p_k^-\) through the
configured Lie-group mean model.

`INEKF-P2`

$$
\Phi_k = \Phi(\Delta t),
\quad
Q_k = \operatorname{diag}(q)\max(\Delta t, 10^{-3}).
$$

`INEKF-P3`

$$
P_k^- = \Phi_k P_{k-1}\Phi_k^\top + Q_k,
\quad
X_k^- =
\begin{bmatrix}
R_k^- & v_k^- & p_k^- \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

The implementation symmetrizes \(P_k^-\) after propagation.

`INEKF-U1`

$$
y_k = z_k - p_k^-[\mathrm{indices}].
$$

Optional Euclidean and Mahalanobis gates can reject the measurement before
correction.

`INEKF-U2`

$$
S_k = H_k P_k^- H_k^\top + R_k,
\quad
K_k = P_k^- H_k^\top S_k^{-1}.
$$

The linearized error covariance uses the same Joseph-form covariance update as
the KF helper.

`INEKF-U3`

$$
\delta_k = K_k y_k,
\quad
dX_k = \exp_{SE_2(3)}(\delta_k),
\quad
X_k = dX_k X_k^-.
$$

Then \(R_k\), \(v_k\), and \(p_k\) are extracted from \(X_k\), and \(P_k\) is
symmetrized.

`INEKF-S1` runs Lie-group prediction, `INEKF-S2` runs position correction, and
`INEKF-S3` returns the benchmark pose vector.

## Invariant EKF 15D

`filters/invariant_kalman_filter_15D.py` keeps the same \(SE_2(3)\) nominal
state but expands the error state to match the common IMU InEKF form used by
the external C++ repositories:

$$
\delta x =
\begin{bmatrix}
\delta\phi \\
\delta v \\
\delta p \\
\delta b_g \\
\delta b_a
\end{bmatrix}
\in \mathbb{R}^{15}.
$$

The nominal state contains:

$$
R,\quad v,\quad p,\quad b_g,\quad b_a.
$$

`INEKF15-P1`

Bias-correct IMU measurements:

$$
\omega_k^c = \omega_k - b_g,
\quad
a_k^c = a_k - b_a.
$$

Then propagate the \(SE_2(3)\) mean with the same gamma functions used by the
external invariant-ekf style propagation:

$$
R_k^- = R_{k-1}\Gamma_0(\omega_k^c \Delta t),
$$

$$
v_k^- =
v_{k-1}
+ R_{k-1}\Gamma_1(\omega_k^c \Delta t)a_k^c\Delta t
+ g\Delta t,
$$

$$
p_k^- =
p_{k-1}
+ v_{k-1}\Delta t
+ R_{k-1}\Gamma_2(\omega_k^c \Delta t)a_k^c\Delta t^2
+ \frac{1}{2}g\Delta t^2.
$$

`INEKF15-P2`

Build the continuous-time 15D error dynamics matrix:

$$
A =
\begin{bmatrix}
-[\omega]^{}_\times & 0 & 0 & -I & 0 \\
-[a]^{}_\times & -[\omega]^{}_\times & 0 & 0 & -I \\
0 & I & -[\omega]^{}_\times & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}.
$$

The discrete transition is:

$$
\Phi_k = \exp(A\Delta t).
$$

`INEKF15-P3`

Propagate covariance:

$$
P_k^- =
\Phi_k P_{k-1}\Phi_k^\top
+ \Phi_k Q_k\Phi_k^\top\Delta t.
$$

`INEKF15-U1`

Use position innovation:

$$
y_k = z_k - p_k^-.
$$

Optional Euclidean and Mahalanobis gates can reject the position measurement.

`INEKF15-U2`

The position measurement matrix selects the \(\delta p\) block:

$$
H =
\begin{bmatrix}
0_{3\times3} & 0_{3\times3} & I_3 & 0_{3\times3} & 0_{3\times3}
\end{bmatrix}.
$$

Then compute:

$$
S_k = H P_k^- H^\top + R_k,
\quad
K_k = P_k^- H^\top S_k^{-1}.
$$

`INEKF15-U3`

Inject the first 9 correction components through \(SE_2(3)\):

$$
\delta_k = K_k y_k,
\quad
dX_k = \exp_{SE_2(3)}(\delta_k[0:9]),
\quad
X_k = dX_k X_k^-.
$$

Then update bias states additively:

$$
b_g \leftarrow b_g + \delta_k[9:12],
\quad
b_a \leftarrow b_a + \delta_k[12:15].
$$

`INEKF15-S1` runs 15D IMU prediction, `INEKF15-S2` runs position correction,
and `INEKF15-S3` returns the benchmark pose vector.
