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


class Rosenbrock(BenchmarkFunction):
    def __init__(self, lower_bound, upper_bound):
        super().__init__(lower_bound, upper_bound)
        
        self.global_optima = np.array([
            [1.0],
            [1.0],
            [0.0]
        ])

        return

    def function(self, pop_array, a = 1.0, b = 100.0, meshgrid = False):

        x, y = np.meshgrid(pop_array[0], pop_array[1]) if meshgrid else (pop_array[0], pop_array[1])
        
        fun = np.square(a - x) + b*np.square(y - np.square(x))

        return fun

    def visualize_surface(self, fig_size = (800, 600), opacity = 0.7, color = 'HSV'):

        x, y = np.meshgrid(
            np.linspace(self.lower_bound[0], self.upper_bound[0]), 
            np.linspace(self.lower_bound[1], self.upper_bound[1])
        )

        z = self.function(np.array([x, y]), meshgrid = False)

        # Draw surface
        fig = go.Figure(
            data = [
                go.Surface(
                z=z, x=x, y=y,
                opacity = opacity,
                colorscale = color,
                name = 'Rosenbrock Function'
                )
            ]
        )

        fig.update_layout(
            template = "plotly_white",
            title = 'Rosenbrock Function', 
            autosize = False,
            width = fig_size[0], height = fig_size[1],

            margin = {
                "l" : 10, 
                "r" : 10, 
                "b" : 50, 
                "t" : 90
            },

            legend={
                "x" : 0, 
                "y" : 0,
                "traceorder" : 'normal',
                "font" : {"size" : 16}
            }
        )

        fig.update_traces(showscale = False)

        # Draw global optima
        fig.add_scatter3d(
            x = self.global_optima[0], y = self.global_optima[1], z = self.global_optima[2],
            mode = 'markers',      
            marker = {
                "symbol" : 'circle',
                "color" : 'rgb(255, 255, 255)',
                "size" : 4
            },
            name = 'Global Optima'
        )

        fig.show()

        return


    def visualize_search(self):

        return

if __name__ == '__main__':

    rosen = Rosenbrock(
        lower_bound = np.array( [1.5, 2.5] ),
        upper_bound = np.array( [-1.5, -0.5] )
    )

    rosen.visualize_surface(
        opacity = 1.0,
        color = "Thermal"
    )