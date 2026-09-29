# 3. Dynamics

How the robot moves, and what the motors can do. This is the model behind the simulation checks, the LQR design and the PID tuning
([05-controllers.md](05-controllers.md)). Code: [lqr_control/model.py](../src/Balance_Car_RL/car/lqr_control/model.py) builds the linear
model from the URDF and `car_cfg.py`; the simulator itself is PhysX.

## 3.1 Symbols and frames

| Symbol | Meaning | Value |
|---|---|---|
| $m$, $L$, $I_b$ | body mass, COM height above the axle, body inertia about the pitch axis through its COM | 0.27 kg, 19.9 mm, $2.17\times10^{-4}$ kg m² |
| $M_w$, $I_w$ | mass and spin inertia of both wheels | 0.04 kg, $3.1\times10^{-5}$ kg m² |
| $J_r$ | rotor inertia of both motors, reflected to the wheels | $3.08\times10^{-4}$ kg m² (2 x 1.54e-4) |
| $r$ | wheel radius | 0.040 m |
| $\theta$ | pitch, positive leaning forward | rad |
| $\psi$ | absolute wheel angle, positive rolls forward | rad |
| $q_L,q_R$ | wheel joint angles as Isaac Lab reports them, relative to the body: $\dot q=\dot\psi-\dot\theta$ | rad |
| $\tau$, $u$ | torque of one wheel, total wheel torque $u=\tau_L+\tau_R$ | N m |

$J_r=J_{rotor}\,G^2$ with $G=\tfrac{3344}{65}\cdot\tfrac{49}{17}=148.3$: as large as the body inertia, so it cannot be left out.

## 3.2 Wheeled inverted pendulum

Rolling without slipping ($x=r\psi$) and the rotor turning at $G(\dot\psi-\dot\theta)$:

$$T=\tfrac12 m\big[(r\dot\psi+L\dot\theta\cos\theta)^2+(L\dot\theta\sin\theta)^2\big]+\tfrac12 I_b\dot\theta^2+\tfrac12 M_w r^2\dot\psi^2+\tfrac12 I_w\dot\psi^2+\tfrac12 J_r(\dot\psi-\dot\theta)^2,\qquad V=mgL\cos\theta$$

Lagrange's equations with generalized forces $u$ on $\psi$ and $-u$ on $\theta$ (the motor pushes the wheel forward and the body
backward):

$$M(\theta)\ddot q + c(\theta,\dot\theta) - \begin{bmatrix}0\\ mgL\sin\theta\end{bmatrix}=\begin{bmatrix}1\\-1\end{bmatrix}u,\qquad q=\begin{bmatrix}\psi\\ \theta\end{bmatrix}$$

Linearized about upright ($\sin\theta\approx\theta$, $\cos\theta\approx1$, no $\dot\theta^2$ terms):

$$M=\begin{bmatrix}(m+M_w)r^2+I_w+J_r & mrL-J_r\\ mrL-J_r & mL^2+I_b+J_r\end{bmatrix}=\begin{bmatrix}8.35\times10^{-4}&-9.30\times10^{-5}\\-9.30\times10^{-5}&6.32\times10^{-4}\end{bmatrix}$$

$\psi$ never appears on the right side, so the state is $z=[\theta,\dot\theta,\dot\psi]$ and $\dot z=Az+Bu$:

$$A=\begin{bmatrix}0&1&0\\ 84.76&0&0\\ 9.44&0&0\end{bmatrix},\qquad B=\begin{bmatrix}0\\-1429.4\\ 1038.6\end{bmatrix}$$

The first column of $A$ is $M^{-1}[0,\ mgL]^\top$, the gravity pull.

| Result | Value |
|---|---|
| Open-loop poles | $0,\ \pm\sqrt{84.76}=\pm9.21$ rad/s: a fall grows by $e$ every 0.11 s |
| Without the $J_r$ terms | 14.9 rad/s |
| Torque authority | gravity torque at 0.25 rad is $mgL\sin0.25=0.013$ N m; the motors give up to $2\tau_{stall}=0.418$ N m |

Because $\psi$ is cyclic, a controller on $z$ regulates the wheel **speed**, not the position: the robot can drift slowly. A lean
is a request for acceleration (a robot that holds a pitch $\theta$ accelerates at about $g\tan\theta$), which is how the cascade of
[04-training.md](04-training.md) makes it drive.

## 3.3 Actuator

Every controller outputs a normalized torque $a\in[-1,1]$ per wheel:

$$\tau_{cmd}=a\,\tau_{stall},\qquad u=2a\,\tau_{stall}\ \text{(both wheels equal)},\qquad a=\operatorname{clip}\!\Big(\frac{u}{2\tau_{stall}},-1,1\Big)$$

The motor cannot deliver the commanded torque at any speed. `DCMotorCfg` clips it with the linear torque-speed line
($\omega$ = wheel speed):

$$\tau\in\Big[\max\big(-\tau_{stall},\ \tau_{stall}(-1-\tfrac{\omega}{\omega_{nl}})\big),\ \min\big(\tau_{stall},\ \tau_{stall}(1-\tfrac{\omega}{\omega_{nl}})\big)\Big]$$

| Constant | Formula | Value |
|---|---|---|
| $\tau_{stall}$ | $0.74\ \text{kg cm}\times0.0980665\times\tfrac{49}{17}$ | 0.209 N m |
| $\omega_{nl}$ | $650\ \text{rpm}\cdot\tfrac{2\pi}{60}\cdot\tfrac{17}{49}$ | 23.6 rad/s (0.94 m/s at the wheel rim) |
| armature | $J_{rotor}G^2$ per wheel | $1.54\times10^{-4}$ kg m² |

The faster a wheel already spins, the less torque is left to accelerate it: this limits the top speed and the recovery from a hard push.

## 3.4 Discretization

The policy runs at 50 Hz over a 200 Hz physics loop and holds the torque for one 20 ms step, so the exact zero-order-hold model is

$$z_{k+1}=A_dz_k+B_du_k,\quad A_d=e^{A\Delta t},\quad B_d=\int_0^{\Delta t}e^{As}B\,ds,\quad \Delta t=0.02$$

$$A_d=\begin{bmatrix}1.0170&0.02011&0\\1.7049&1.0170&0\\0.1898&0.001893&1\end{bmatrix},\qquad B_d=\begin{bmatrix}-0.287\\-28.75\\ 20.75\end{bmatrix}$$

## 3.5 What the model leaves out

Gearbox friction and backlash, motor electrical lag, battery sag, tyre slip, and the yaw dynamics of two wheels (the baselines
command both wheels equally). See [01-robot.md](01-robot.md) §1.5.
