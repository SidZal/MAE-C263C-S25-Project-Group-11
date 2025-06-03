import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, Diagram, Context
from pydrake.systems.primitives import MatrixGain, LogVectorOutput

from mechae263C_helpers.drake import LinearCombination, plot_diagram
from mechae263C_helpers.hw4.arm import Arm
from mechae263C_helpers.hw4.kinematics import calc_fk_2D
from mechae263C_helpers.hw4.trajectory import (
    JointSpaceTrajectorySource,
)


def run_simulation(
    q_initial: NDArray[np.double],
    q_final: NDArray[np.double],
    B_avg: NDArray[np.double],
    K_p: NDArray[np.double],
    K_d: NDArray[np.double],
    simulation_duration_s: float,
    should_apply_control_torques: bool,
    control_period_s: float = 1e-3,
) -> tuple[
    NDArray[np.double],
    tuple[NDArray[np.double], NDArray[np.double]],
    tuple[NDArray[np.double], NDArray[np.double]],
    NDArray[np.double],
    Diagram,
]:
    """
    Runs a simulation with a desired joint position

    Parameters
    ----------
    q_initial:
        A numpy array of shape (2,) containing the initial joint positions

    q_final:
        A numpy array of shape (2,) containing the final desired joint positions

    B_avg:
        A numpy array of shape (2, 2) containing the average linearized inertia matrix

    K_p:
        A numpy array of shape (2, 2) containing the proportional gains of the inverse
        dynamics controller.

    K_d:
        A numpy array of shape (2, 2) containing the derivative gains of the inverse
        dynamics controller.

    control_period_s:
        The period between control commands in units of seconds

    simulation_duration_s:
        The duration of the simulation in units of seconds

    should_apply_control_torques:
        A bool that specifies that control torques should be simulated when set to
        `True`. (If set to `False` then no control torques are simulated).

    Returns
    -------
    A tuple with five elements:
        1. A numpy array with shape (T,) of simulation time steps
        2. A tuple of numpy arrays both with shape (2, T) of desired and actual joint
           positions corresponding to each simulation time step, respectively.
        3. A tuple of numpy arrays both with shape (2, T) of desired and actual joint
           velocities corresponding to each simulation time step, respectively.
        4. A numpy array with shape (2, T) of applied control torques corresponding to
           each simulation time step
        5. A Drake diagram
    """
    # ----------------------------------------------------------------------------------
    # Add "systems" to a `DiagramBuilder` object.
    #   - "systems" are the blocks in a block diagram
    #   - Some examples for how to add named systems to a `DiagramBuilder` are given
    #     below
    # ----------------------------------------------------------------------------------
    builder = DiagramBuilder()

    # Create the desired joint angle, velocity, and acceleration trajectories
    dt = control_period_s
    times = np.arange(0, simulation_duration_s + dt, dt)
    waypoint_times = np.asarray([0, simulation_duration_s / 2, simulation_duration_s])
    waypoints = np.stack([q_initial, np.deg2rad([130, -110]), q_final], axis=1)

    q_d_traj, qdot_d_traj, qddot_d_traj = eval_cubic_spline_traj(
        times=times, waypoint_times=waypoint_times, waypoints=waypoints
    )
    q_traj = builder.AddNamedSystem(
        "q_d_traj",
        JointSpaceTrajectorySource(
            name="q_d_traj",
            num_joints=q_d_traj.shape[0],
            times=times,
            joint_coordinates=q_d_traj,
        ),
    )
    qdot_traj = builder.AddNamedSystem(
        "qdot_d_traj",
        JointSpaceTrajectorySource(
            name="qdot_d_traj",
            num_joints=qdot_d_traj.shape[0],
            times=times,
            joint_coordinates=qdot_d_traj,
        ),
    )
    if should_apply_control_torques:
        qddot_traj = builder.AddNamedSystem(
            "qddot_d_traj",
            JointSpaceTrajectorySource(
                name="qddot_d_traj",
                num_joints=qddot_d_traj.shape[0],
                times=times,
                joint_coordinates=qddot_d_traj,
            ),
        )

        K_p_gain = builder.AddNamedSystem(
            "K_p", MatrixGain(np.asarray(K_p, dtype=np.double))
        )
        K_d_gain = builder.AddNamedSystem(
            "K_d", MatrixGain(np.asarray(K_d, dtype=np.double))
        )

    joint_position_error = builder.AddNamedSystem(
        "joint_position_error",
        LinearCombination(input_coeffs=(1, -1), input_shapes=(2,)),
    )
    joint_velocity_error = builder.AddNamedSystem(
        "joint_velocity_error",
        LinearCombination(input_coeffs=(1, -1), input_shapes=(2,)),
    )
    arm = builder.AddNamedSystem("arm", Arm())

    if should_apply_control_torques:
        control_torque = builder.AddNamedSystem(
            "u", LinearCombination(input_coeffs=(1, 1, 1), input_shapes=(2,))
        )
        inertia_matrix = builder.AddNamedSystem("B_avg", MatrixGain(B_avg))

    # ----------------------------------------------------------------------------------
    # Connect the systems in the `DiagramBuilder` (i.e. add arrows of block diagram)
    # ----------------------------------------------------------------------------------
    # `builder.ExportInput(input_port)` makes the provided "input_port" into an input
    # of the entire diagram
    # The functions system.get_input_port() returns the input port of the given system
    #   - If there is more than one input port, you must specify the index of the
    #     desired input
    # The functions system.get_output_port() returns the output port of the given system
    #   - If there is more than one output port, you must specify the index of the
    #     desired output
    builder.Connect(q_traj.get_output_port(), joint_position_error.get_input_port(0))
    builder.Connect(qdot_traj.get_output_port(), joint_velocity_error.get_input_port(0))
    if should_apply_control_torques:
        builder.Connect(qddot_traj.get_output_port(), inertia_matrix.get_input_port())

    joint_velocity_output = arm.get_output_port(0)
    joint_position_output = arm.get_output_port(1)

    # TODO:
    #   Replace any `...` below with the correct system and values. Please keep the
    #   system names the same
    builder.Connect(joint_position_output, joint_position_error.get_input_port(1))
    builder.Connect(joint_velocity_output, joint_velocity_error.get_input_port(1))
    if should_apply_control_torques:
        builder.Connect(joint_position_error.get_output_port(), K_p_gain.get_input_port())
        builder.Connect(joint_velocity_error.get_output_port(), K_d_gain.get_input_port())

        #
        builder.Connect(
            inertia_matrix.get_output_port(), control_torque.get_input_port(0)
        )
        builder.Connect(K_p_gain.get_output_port(), control_torque.get_input_port(1))
        builder.Connect(K_d_gain.get_output_port(), control_torque.get_input_port(2))
        builder.Connect(control_torque.get_output_port(), arm.get_input_port())
    else:
        builder.ExportInput(arm.get_input_port(), name="control_torque")

    # ----------------------------------------------------------------------------------
    # Log joint positions
    # ----------------------------------------------------------------------------------
    # These systems are special in Drake. They periodically save the output port value
    # a during a simulation so that it can be accessed later. The value is saved every
    # `publish_period` seconds in simulation time.
    joint_position_logger = LogVectorOutput(
        arm.get_output_port(1), builder, publish_period=control_period_s
    )
    joint_velocity_logger = LogVectorOutput(
        arm.get_output_port(0), builder, publish_period=control_period_s
    )
    if should_apply_control_torques:
        control_torque_logger = LogVectorOutput(
            control_torque.get_output_port(), builder, publish_period=control_period_s
        )

    # ----------------------------------------------------------------------------------
    # Setup/Run the simulation
    # ----------------------------------------------------------------------------------
    # This line builds a `Diagram` object and uses it to make a `Simulator` object for
    # the diagram
    diagram: Diagram = builder.Build()
    diagram.set_name("Inverse Dynamics Controller")
    simulator: Simulator = Simulator(diagram)

    # Get the context (this contains all the information needed to run the simulation)
    context: Context = simulator.get_mutable_context()

    # Set initial conditions
    initial_conditions = context.get_mutable_continuous_state_vector()
    initial_conditions.SetAtIndex(2, q_initial[0])
    initial_conditions.SetAtIndex(3, q_initial[1])

    # Advance the simulation by `simulation_duration_s` seconds using the
    # `simulator.AdvanceTo()` function
    simulator.AdvanceTo(simulation_duration_s)

    # ----------------------------------------------------------------------------------
    # Extract simulation outputs
    # ----------------------------------------------------------------------------------
    # The lines below extract the joint position log from the simulator context
    joint_position_log = joint_position_logger.FindLog(simulator.get_context())
    t = joint_position_log.sample_times()
    q_actual = joint_position_log.data()

    joint_velocity_log = joint_velocity_logger.FindLog(simulator.get_context())
    qdot_actual = joint_velocity_log.data()

    control_torques = np.zeros((2, len(t)), dtype=np.double)

    if should_apply_control_torques:
        control_torque_log = control_torque_logger.FindLog(simulator.get_context())
        control_torques = control_torque_log.data()

    # Return a `tuple` of required results
    return t, (q_d_traj, q_actual), (qdot_d_traj, qdot_actual), control_torques, diagram



    fig, _ = plot_diagram(diagram)
    fig.savefig("drake_diagram.png")
