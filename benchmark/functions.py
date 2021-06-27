import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo


class BenchmarkFunction:
    def __init__(self, lower_bound, upper_bound):

        self.global_optima = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def function(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

    def draw_suface(self, x, y, z, opacity, name, fig_size):
        # Draw surface
        surface = go.Figure(
            data=[
                go.Surface(
                    z=z,
                    x=x,
                    y=y,
                    opacity=opacity,
                    colorscale="Spectral",
                    name=name,
                )
            ]
            * 3
        )

        # Update layout
        surface.update_layout(
            template="plotly_white",
            title=name,
            autosize=False,
            width=fig_size[0],
            height=fig_size[1],
            margin={"l": 10, "r": 10, "b": 50, "t": 90},
            legend={"x": 0, "y": 0, "traceorder": "normal", "font": {"size": 16}},
        )

        surface.update_traces(showscale=False)

        return surface

    def create_animation_buttons(self, duration):

        buttons = [
            {
                "type": "buttons",
                "direction": "left",
                "pad": {"b": 10, "t": 60},
                "type": "buttons",
                "x": 0.1,
                "y": 0.2,
                "buttons": [
                    # Play button
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": duration},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition": {
                                    "duration": duration,
                                    "easing": "linear",
                                },
                            },
                        ],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    # Pause button
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                        "label": "Pause;",  # pause symbol
                        "method": "animate",
                    },
                ],
            }
        ]

        return buttons

    def create_animation_frames(self, pop_array, best_agents_array, z_offset=2.5):

        # Number of iterations
        t_max = len(pop_array)

        frames = [
            go.Frame(
                name = str(t),
                data=[
                    # Population array for iteration t
                    go.Scatter3d(
                        x=pop_array[t][0, :],
                        y=pop_array[t][1, :],
                        z=pop_array[t][2, :] + z_offset,
                        mode="markers",
                        marker=dict(symbol="circle", color="rgb(12, 200, 17)", size=4),
                        name="Search agents",
                    ),
                    # Best agents array for iteration t
                    go.Scatter3d(
                        x=best_agents_array[t][0, :],
                        y=best_agents_array[t][1, :],
                        z=best_agents_array[t][2, :] + z_offset,
                        mode="markers",
                        marker=dict(
                            symbol="diamond", color="rgb(220, 221, 28)", size=4
                        ),
                        name="Best agents",
                    ),
                ],
                traces=[1, 2],
            )
            for t in range(t_max)
        ]

        return frames

    def create_animation_sliders(self, frames):
        sliders = [
            {
                "active": 0,
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Iteration:",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(frames)
                ],
            }
        ]

        return sliders


class Rosenbrock(BenchmarkFunction):
    def __init__(self, lower_bound, upper_bound):
        super().__init__(lower_bound, upper_bound)

        """
        lower_bound:
            - shape: (dim, 1)
        upper_bound:
            - shape: (dim, 1)
        
        """

        self.global_optima = np.array([[1.0], [1.0], [0.0]])

        return

    def function(self, pop_array, a=1.0, b=100.0, meshgrid=False):

        x, y = (
            np.meshgrid(pop_array[0], pop_array[1])
            if meshgrid
            else (pop_array[0], pop_array[1])
        )

        fun = np.square(a - x) + b * np.square(y - np.square(x))

        return fun

    def visualize_surface(
        self,
        fig_size=(800, 600),
        opacity=0.7,
        color="HSV",
        using_notebook=False,
    ):
        # Enable Jupyter Notebook rendering
        # if using_notebook:
        #     pyo.init_notebook_mode(connected=False)
        #     pio.renderers.default = "notebook"

        x, y = np.meshgrid(
            np.linspace(self.lower_bound[0], self.upper_bound[0]),
            np.linspace(self.lower_bound[1], self.upper_bound[1]),
        )

        z = self.function(np.array([x, y]), meshgrid=False)

        # Draw surface
        surface = self.draw_suface(x, y, z, opacity, "Rosenbrock Function", fig_size)

        # Draw global optima
        surface.add_scatter3d(
            x=self.global_optima[0],
            y=self.global_optima[1],
            z=self.global_optima[2],
            mode="markers",
            marker={"symbol": "circle", "color": "rgb(255, 255, 255)", "size": 4},
            name="Global Optima",
        )

        surface.show()

        return

    def visualize_search(
        self,
        pop_array,
        best_agents_array,
        fig_size=(800, 600),
        opacity=0.7,
        color="HSV",
        using_notebook=False,
        frame_duration=1000,  # ms
    ):
        # Enable Jupyter Notebook rendering
        # if using_notebook:
        #     pyo.init_notebook_mode(connected=False)
        #     pio.renderers.default = "notebook"

        x, y = np.meshgrid(
            np.linspace(self.lower_bound[0], self.upper_bound[0]),
            np.linspace(self.lower_bound[1], self.upper_bound[1]),
        )
        z = self.function(np.array([x, y]), meshgrid=False)

        # Draw surface
        surface = self.draw_suface(
            x, y, z, opacity, "Search on Rosenbrock Function", fig_size
        )

        # Draw the global optima
        surface.add_scatter3d(
            x=self.global_optima[0],
            y=self.global_optima[1],
            z=self.global_optima[2],
            mode="markers",
            marker={"symbol": "circle", "color": "rgb(255, 255, 255)", "size": 4},
            name="Global Optima",
        )

        # Create the animation frames
        frames = self.create_animation_frames(pop_array, best_agents_array)

        # Create animation buttons
        buttons = self.create_animation_buttons(frame_duration)

        # Create animation sliders
        sliders = self.create_animation_sliders(frames)

        # Add everything to the surface graph
        surface.frames = frames
        surface.layout.updatemenus = buttons
        surface.layout.sliders = sliders

        surface.show()

        return


if __name__ == "__main__":

    lower_bound = np.array([1.5, 2.5])
    upper_bound = np.array([-1.5, -0.5])

    rosen = Rosenbrock(lower_bound=lower_bound, upper_bound=upper_bound)

    rosen.visualize_surface(opacity=1.0, color="Thermal")
