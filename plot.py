import manim as M
import pandas as pd
import numpy as np


class Plot(M.MovingCameraScene):
    def __init__(self):
        M.config.frame_width = 16

        # Path of the files
        signal_path = "data/signal.csv"
        features_path = "data/features.csv"

        # Loading all the required data
        self.signal_frame = pd.read_csv(signal_path)
        self.features_frame = pd.read_csv(features_path)
        self.times = self.signal_frame["time"].values
        self.lead1 = self.signal_frame["lead1"].values
        self.lead2 = self.signal_frame["lead2"].values
        self.beat_idx = self.features_frame["position"].values
        self.dss = self.features_frame["dss"].values
        self.dss_lag = self.features_frame["dss_lag"].values

        # NOTE: Change the constants below as required

        # Time in seconds to create one cycle
        self.CYCLE_CREATION_RUNTIME = 1
        # A float between 0 and 1 determining the delay in
        # creation of a cycle and removal of previous cycle
        self.DELAY_BETWEEN_CYCLES = 0.1
        # Radius of the mean dss circle
        self.MEAN_DSS_RADIUS = 0.25
        # color scheme of the whole scene
        self.color_scheme = {
            "bg": M.BLACK,
            "axes": M.GREY,
            "graph_creation": M.BLUE,
            "graph_removal": M.TEAL,
            "beats": M.YELLOW,
            "ss_line": M.WHITE,
            "dss_line": M.GREEN,
            "mean_dss_circle": M.BLUE,
            "healthy": M.WHITE,
            "non-healthy": M.RED,
        }
        M.config.background_color = self.color_scheme["bg"]

        super().__init__()

    def setup_axes(self):
        """[Add the required axes to the scene]"""

        def get_grid(
            axes,
            color=self.color_scheme["axes"],
            stroke_width=0.2,
        ):
            """[Given an axes it returns a lines of a grid]

            Returns:
                [VDict]: [Return a VDict containing the grid lines]
            """
            vertical_lines = M.VGroup()
            horizontal_lines = M.VGroup()

            x_start, x_end, x_step = axes.x_range
            y_start, y_end, y_step = axes.y_range

            for x in np.arange(x_start + x_step, x_end, x_step):
                start_point = axes.coords_to_point(x, y_start)
                end_point = axes.coords_to_point(x, y_end)
                line = M.Line(start_point, end_point).set_stroke(
                    color=color, width=stroke_width
                )
                vertical_lines.add(line)

            for y in np.arange(y_start + y_step, y_end, y_step):
                start_point = axes.coords_to_point(x_start, y)
                end_point = axes.coords_to_point(x_end, y)
                line = M.Line(start_point, end_point).set_stroke(
                    color=color, width=stroke_width
                )
                horizontal_lines.add(line)

            mappings = [
                ("vertical_lines", vertical_lines),
                ("horizontal_lines", horizontal_lines),
            ]
            grid = M.VDict(mappings)

            return grid

        # Axes for data of lead1
        self.lead1_axes = (
            M.Axes(
                x_range=[0, 10, 1],
                y_range=[-0.2, 0.5, 0.1],
                x_length=7.5,
                y_length=3,
                axis_config={"include_tip": False, "number_scale_value": 0.3},
                x_axis_config={
                    "numbers_to_include": np.arange(0, 10 + 1, 1),
                },
                y_axis_config={
                    "decimal_number_config": {"num_decimal_places": 1},
                    "numbers_to_include": np.arange(-0.2, 0.51, 0.1),
                    "numbers_to_exclude": [],
                },
                tips=False,
            )
            .set_color(self.color_scheme["axes"])
            .to_edge(M.UL)
        )

        # Axes for data of lead2
        self.lead2_axes = (
            M.Axes(
                x_range=[0, 10, 1],
                y_range=[-0.5, 0.2, 0.1],
                x_length=7.5,
                y_length=3,
                axis_config={"include_tip": False, "number_scale_value": 0.3},
                x_axis_config={
                    "numbers_to_include": np.arange(0, 10 + 1, 1),
                },
                y_axis_config={
                    "decimal_number_config": {"num_decimal_places": 1},
                    "numbers_to_include": np.arange(-0.5, 0.21, 0.1),
                    "numbers_to_exclude": [],
                },
                tips=False,
            )
            .set_color(self.color_scheme["axes"])
            .to_edge(M.DL)
        )

        # Axes for the scatter graph
        self.scatter_axes = (
            M.Axes(
                x_range=[-1, 1, 0.25],
                y_range=[-1, 1, 0.25],
                x_length=6,
                y_length=6,
                axis_config={
                    "include_tip": False,
                    "number_scale_value": 0.3,
                    "decimal_number_config": {"num_decimal_places": 2},
                    "numbers_to_exclude": [0],
                },
                x_axis_config={
                    "numbers_to_include": np.arange(-1, 1.25, 0.25),
                },
                y_axis_config={
                    "numbers_to_include": np.arange(-1, 1.25, 0.25),
                },
                tips=False,
            )
            .set_color(self.color_scheme["axes"])
            .to_edge(M.RIGHT)
        )

        # Add the labels on the graph
        scatter_axes_legend = self.get_scatter_axes_legend(self.scatter_axes)

        # Adding the axes to the scene
        self.add(self.lead1_axes)
        self.add(get_grid(self.lead1_axes))
        self.add(self.lead2_axes)
        self.add(get_grid(self.lead2_axes))
        self.add(self.scatter_axes)
        self.add(get_grid(self.scatter_axes))
        self.add(scatter_axes_legend)

    def get_scatter_axes_legend(self, axes):
        legend = M.VGroup()
        mean_dss_label = M.VGroup()
        healthy_label = M.VGroup()
        non_healthy_label = M.VGroup()

        mean_dss_rect = M.Rectangle(self.color_scheme["mean_dss_circle"])
        mean_dss_rect.scale(0.5)
        mean_dss_text = M.Text("mean dss distance").next_to(mean_dss_rect)
        mean_dss_label.add(mean_dss_rect, mean_dss_text)

        healthy_line = M.Dot(color=self.color_scheme["healthy"], radius=0.3)
        healthy_text = M.Text("healthy").next_to(healthy_line)
        healthy_label.add(healthy_line, healthy_text)

        non_healthy_line = M.Dot(
            color=self.color_scheme["non-healthy"], radius=0.3
        )
        non_healthy_text = M.Text("non-healthy").next_to(non_healthy_line)
        non_healthy_label.add(non_healthy_line, non_healthy_text)

        legend.add(mean_dss_label, healthy_label, non_healthy_label)
        legend.arrange(direction=M.DOWN, aligned_edge=M.LEFT, buff=1)

        return legend.scale(0.25).to_edge(M.UR)

    def setup_points(self):
        """[Sets up all the required data points for the scene]"""
        lead1_axes_x_max = self.lead1_axes.x_range[1]
        self.lead1_all_points = [
            self.lead1_axes.coords_to_point(
                self.times[i] % (lead1_axes_x_max), self.lead1[i]
            )
            for i in range(len(self.times))
        ]

        lead2_axes_x_max = self.lead2_axes.x_range[1]
        self.lead2_all_points = [
            self.lead2_axes.coords_to_point(
                self.times[i] % (lead2_axes_x_max), self.lead2[i]
            )
            for i in range(len(self.times))
        ]

        self.beats = [
            M.Dot(
                self.lead1_axes.coords_to_point(
                    self.times[i] % (lead2_axes_x_max), 0.5
                )
            )
            if i in self.beat_idx
            else None
            for i in range(len(self.times))
        ]

        self.scatter_all_points = []
        # Populating the scatter_all_points list
        for i in range(len(self.times)):
            if i in self.beat_idx:
                idx = list(self.beat_idx).index(i)
                if self.dss[idx] != np.NaN and self.dss_lag[idx] != np.NaN:
                    self.scatter_all_points.append(
                        M.Dot(
                            self.scatter_axes.coords_to_point(
                                self.dss[idx], self.dss_lag[idx]
                            ),
                            radius=0.05,
                        )
                    )
                else:
                    self.scatter_all_points.append(None)
            else:
                self.scatter_all_points.append(None)

        # An array containing indexes where the graph should wrap around
        # along with the start and end index
        self.checkpoints = [0]
        num_lines = 0
        for i in range(1, len(self.times)):
            if self.times[i] % (lead2_axes_x_max) != 0:
                num_lines += 1
            else:
                self.checkpoints.append(i - 1)
        self.checkpoints = self.checkpoints + [num_lines]

    def get_line(self, dot1, dot2):
        return M.DoubleArrow(
            dot1.get_center(),
            dot2.get_center(),
            buff=0,
            max_tip_length_to_length_ratio=0.15,
            max_stroke_width_to_length_ratio=2,
        ).set_color(self.color_scheme["ss_line"])

    def animate_ss(self, ss_lines):
        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5),"
            a = iter(iterable)
            return zip(a, a)

        dss_lines = []
        for pair in pairwise(ss_lines):
            pair = sorted(pair, key=lambda obj: obj.get_length())

            # Arranging consecutive ss_lines on top of each other
            self.play(pair[0].animate.shift(0.2 * M.UP))
            self.play(
                pair[1]
                .animate.next_to(
                    pair[0],
                    direction=M.ORIGIN,
                    buff=0,
                    aligned_edge=M.LEFT,
                )
                .shift(0.1 * M.UP)
            )

            # The dss line
            x1 = pair[0].get_edge_center(direction=M.RIGHT)[0]
            y1 = pair[0].get_edge_center(direction=M.RIGHT)[1]
            x2 = pair[1].get_edge_center(direction=M.RIGHT)[0]
            start = np.array([x1, y1, 0])
            end = np.array([x2, y1, 0])
            dss_line = M.DoubleArrow(
                start,
                end,
                color=self.color_scheme["dss_line"],
                buff=0,
                max_stroke_width_to_length_ratio=2,
            )
            dss_lines.append(dss_line)
            self.play(M.Create(dss_line))

        return dss_lines

    def set_scatter_dot_color(self, i, radius, dot):
        idx = list(self.beat_idx).index(i)
        x = self.dss[idx]
        y = self.dss_lag[idx]
        distance = np.sqrt(x * x + y * y)
        if distance <= radius:
            dot.set_color(self.color_scheme["healthy"])
        else:
            dot.set_color(self.color_scheme["non-healthy"])

    def construct(self):
        self.setup_axes()
        self.setup_points()

        # Keeping storage of the cycle for removal from scene later on
        # Used in the inner function (lines_with_beats)
        lead1_cycle = None
        lead2_cycle = None
        beats = None

        def lines_with_beats(start, end, line_color, create=True):
            """[Returns the animations for the graph and the beats]

            Args:
                start ([int]): [index of the checkpoint array to start at]
                end ([int]): [index of the checkpoint array to end at]
                line_color ([string]): [color of the graph]
                create (bool, optional):
                [boolean to determine if graph should be created or removed].
                Defaults to True.

            Returns:
                [list]: [contains the animations to be played out for a cycle]
            """
            animate_func = M.Create if create else M.Uncreate

            if create:
                start_idx = self.checkpoints[start] + 1
                end_idx = self.checkpoints[end]

                lead1_cycle_points = self.lead1_all_points[start_idx:end_idx]
                lead2_cycle_points = self.lead2_all_points[start_idx:end_idx]

                nonlocal lead1_cycle
                nonlocal lead2_cycle
                nonlocal beats

                lead1_cycle = M.VGroup().set_points_smoothly(
                    lead1_cycle_points
                )
                lead2_cycle = M.VGroup().set_points_smoothly(
                    lead2_cycle_points
                )
                beats = M.VGroup()

                for beat in self.beats[start_idx:end_idx]:
                    if beat:
                        beats.add(beat)
            else:
                lead1_cycle.reverse_points()
                lead2_cycle.reverse_points()

            animations = [
                animate_func(lead1_cycle.set_color(line_color)),
                animate_func(lead2_cycle.set_color(line_color)),
            ]

            # Adding the Fade In from Bottom animation of the beats
            if create:
                beats_anims = M.AnimationGroup(
                    *[
                        M.FadeIn(beat, shift=0.5 * M.UP)
                        for beat in beats.set_color(self.color_scheme["beats"])
                    ],
                    lag_ratio=1,
                    run_time=self.CYCLE_CREATION_RUNTIME,
                    rate_func=M.rate_functions.linear,
                )
                animations.append(beats_anims)
            else:
                animations.append(animate_func(beats))

            return animations

        # Animating the creation of the graphs and beats
        for i in range(len(self.checkpoints) - 1):
            if i == 0:
                # Create the first cycle
                self.play(
                    M.AnimationGroup(
                        *lines_with_beats(
                            i, i + 1, self.color_scheme["graph_creation"]
                        ),
                        run_time=self.CYCLE_CREATION_RUNTIME,
                        rate_func=M.rate_functions.linear,
                    )
                )

                # Animation of the first point on the scatter graph
                self.camera.frame.save_state()
                ss_12 = self.get_line(beats[0], beats[1])
                ss_23 = self.get_line(beats[1], beats[2])
                ss_34 = self.get_line(beats[2], beats[3])
                ss_lines = [ss_12, ss_23, ss_23.copy(), ss_34]
                self.play(
                    self.camera.frame.animate.set(
                        width=ss_12.width * 10
                    ).move_to(M.VGroup(*beats[0:4]))
                )
                self.add(*ss_lines)
                # Animation for the difference between ss
                dss_lines = self.animate_ss(ss_lines)
                dss_labels = []
                # labels for the dss lines
                dss_labels.append(
                    M.Text(str(self.dss_lag[3]))
                    .scale(0.1)
                    .next_to(dss_lines[0], direction=M.DOWN, buff=M.SMALL_BUFF/2)
                )
                dss_labels.append(
                    M.Text(str(self.dss[3]))
                    .scale(0.1)
                    .next_to(dss_lines[1], direction=M.DOWN, buff=M.SMALL_BUFF/2)
                )
                self.play(*[M.Write(label) for label in dss_labels])
                self.wait(1)
                self.remove(
                    *ss_lines,
                )

                # Transforming dss to a point on the scatter graph
                scatter_point = self.scatter_axes.coords_to_point(
                    self.dss[3], self.dss_lag[3]
                )
                scatter_dot = M.Dot(scatter_point, radius=0.05)
                self.set_scatter_dot_color(
                    self.beat_idx[i], self.MEAN_DSS_RADIUS, scatter_dot
                )
                # coord label
                scatter_coords = (
                    M.Text(
                        "("
                        + str(self.dss[3])
                        + ", "
                        + str(self.dss_lag[3])
                        + ")"
                    )
                    .scale(0.35)
                    .next_to(scatter_dot)
                )

                mean_dss_circle = self.scatter_axes.get_parametric_curve(
                    lambda theta: np.array(
                        (
                            self.MEAN_DSS_RADIUS * np.cos(theta),
                            self.MEAN_DSS_RADIUS * np.sin(theta),
                            0,
                        )
                    ),
                    t_range=np.array([0, M.TAU]),
                ).set_stroke(
                    color=self.color_scheme["mean_dss_circle"], width=2
                )

                self.play(
                    M.Restore(self.camera.frame),
                    M.ReplacementTransform(M.VGroup(*dss_lines), scatter_dot),
                    M.ReplacementTransform(
                        M.VGroup(*dss_labels), scatter_coords
                    ),
                    M.Create(mean_dss_circle),
                )
                self.play(M.FadeOut(scatter_coords))
            # Creating the next cycle and removing the previous
            else:
                self.play(
                    M.LaggedStart(
                        M.AnimationGroup(
                            *lines_with_beats(
                                i - 1,
                                i,
                                self.color_scheme["graph_removal"],
                                create=False,
                            ),
                        ),
                        M.AnimationGroup(
                            *lines_with_beats(i, i + 1, M.BLUE),
                        ),
                        lag_ratio=self.DELAY_BETWEEN_CYCLES,
                        run_time=2 * self.CYCLE_CREATION_RUNTIME,
                        rate_func=M.rate_functions.linear,
                    )
                )

            # Animate the remaining scatter points
            scatter_points_animations = []
            for j in range(self.checkpoints[i], self.checkpoints[i + 1]):
                scatter_dot = self.scatter_all_points[j]
                if scatter_dot:
                    # setting the health color of the dot
                    self.set_scatter_dot_color(
                        j, self.MEAN_DSS_RADIUS, scatter_dot
                    )
                    scatter_points_animations.append(
                        M.FadeIn(
                            scatter_dot,
                        )
                    )
            self.play(
                M.AnimationGroup(
                    *scatter_points_animations, lag_ratio=0.8, run_time=2
                )
            )

        self.wait()
