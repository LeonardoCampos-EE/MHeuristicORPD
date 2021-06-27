import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo


def rosenbrockFunc(varArray, a=1.0, b=100.0, meshGrid=False):

    if meshGrid:
        x, y = np.meshgrid(varArray[0], varArray[1])
    else:
        x, y = varArray[0], varArray[1]

    f = np.square(a - x) + b * np.square(y - np.square(x))
    return f


def showSurface(
    surfaceArray, globalOptima, figSize=(800, 600), opacity=0.7, color="HSV"
):

    x, y = np.meshgrid(surfaceArray[0], surfaceArray[1])
    z = rosenbrockFunc(surfaceArray, meshGrid=True)

    # Draw surface
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x,
                y=y,
                opacity=opacity,
                colorscale=color,
                name="Rosenbrock Function",
            )
        ]
    )

    w, h = figSize[0], figSize[1]
    fig.update_layout(
        template="plotly_white",
        title="Grey Wolf Optimizer on the Rosenbrock Function",
        autosize=False,
        width=w,
        height=h,
        margin=dict(l=10, r=10, b=50, t=90),
        legend=dict(
            x=0,
            y=0,
            traceorder="normal",
            font=dict(
                size=16,
            ),
        ),
    )
    fig.update_traces(showscale=False)

    # Global optima points
    xOpt, yOpt, zOpt = globalOptima[0], globalOptima[1], globalOptima[2]

    # Draw global optima
    fig.add_scatter3d(
        x=xOpt,
        y=yOpt,
        z=zOpt,
        mode="markers",
        marker=dict(symbol="circle", color="rgb(255, 255, 255)", size=4),
        name="Global Optima",
    )

    fig.show()

    return


def showSearch(
    surfaceArray,
    globalOptima,
    popTensor,
    bestTensor,
    t=0,
    figSize=(800, 600),
    opacity=0.7,
):

    off = 1.5

    # Surface points
    x, y = np.meshgrid(surfaceArray[0], surfaceArray[1])
    z = rosenbrockFunc(surfaceArray, meshGrid=True)

    # Draw surface
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x,
                y=y,
                opacity=opacity,
                colorscale="Spectral",
                name="Rosenbrock Function",
            )
        ]
    )

    w, h = figSize[0], figSize[1]
    fig.update_layout(
        template="plotly_white",
        title="Grey Wolf Optimizer on the Rosenbrock Function",
        autosize=False,
        width=w,
        height=h,
        margin=dict(l=10, r=10, b=50, t=90),
        legend=dict(
            x=0,
            y=0.5,
            traceorder="normal",
            font=dict(
                size=16,
            ),
        ),
    )

    fig.update_traces(showscale=False)

    # Global optima points
    xOpt, yOpt, zOpt = globalOptima[0], globalOptima[1], globalOptima[2]

    # Draw global optima
    fig.add_scatter3d(
        x=xOpt,
        y=yOpt,
        z=zOpt,
        mode="markers",
        marker=dict(symbol="circle", color="rgb(0, 0, 0)", size=4),
        name="Global Optima",
    )

    # Draw population
    xPop, yPop, zPop = popTensor[t][0], popTensor[t][1], popTensor[t][2]

    fig.add_scatter3d(
        x=xPop,
        y=yPop,
        z=zPop + off,
        mode="markers",
        marker=dict(symbol="circle", color="rgb(12, 200, 17)", size=4),
        name="Wolves",
    )

    # Draw best agents
    Alpha = bestTensor[t][:, 0]
    zAlpha = rosenbrockFunc(Alpha)

    Beta = bestTensor[t][:, 1]
    zBeta = rosenbrockFunc(Beta)

    Delta = bestTensor[t][:, 2]
    zDelta = rosenbrockFunc(Delta)

    fig.add_scatter3d(
        x=[Alpha[0]],
        y=[Alpha[1]],
        z=[zAlpha + off],
        mode="markers",
        marker=dict(symbol="diamond", color="rgb(12, 13, 130)", size=4),
        name="Alpha wolf",
    )

    fig.add_scatter3d(
        x=[Beta[0]],
        y=[Beta[1]],
        z=[zBeta + off],
        mode="markers",
        marker=dict(symbol="diamond", color="rgb(220, 221, 28)", size=4),
        name="Beta wolf",
    )

    fig.add_scatter3d(
        x=[Delta[0]],
        y=[Delta[1]],
        z=[zDelta + off],
        mode="markers",
        marker=dict(symbol="diamond", color="rgb(255, 6, 0)", size=4),
        name="Delta wolf",
    )

    fig.show()

    return


def showAnimation(
    surfaceArray,
    globalOptima,
    popTensor,
    bestTensor,
    figSize=(800, 600),
    opacity=0.7,
    notebook=False,
    frameDuration=500,
):

    if notebook:
        pyo.init_notebook_mode(connected=False)
        pio.renderers.default = "notebook"

    # Surface points
    x, y = np.meshgrid(surfaceArray[0], surfaceArray[1])
    z = rosenbrockFunc(surfaceArray, meshGrid=True)

    # Draw surface
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x,
                y=y,
                opacity=opacity,
                colorscale="Spectral",
                name="Função de Rosenbrock",
            )
        ]
        * 5
    )

    fig.update_traces(showscale=False)

    # Global optima points
    xOpt, yOpt, zOpt = globalOptima[0], globalOptima[1], globalOptima[2]
    # Draw global optima
    fig.add_scatter3d(
        x=xOpt,
        y=yOpt,
        z=zOpt,
        mode="markers",
        marker=dict(symbol="circle", color="rgb(0, 0, 0)", size=4),
        name="Ótimo Global",
    )

    # Animation
    fig.frames = createFrames(popTensor, bestTensor)

    """
    sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]
    """
    w, h = figSize[0], figSize[1]
    if notebook:
        xLeg = 0
        yLeg = 0.5
    else:
        xLeg = 0.15
        yLeg = 0.5

    fig.update_layout(
        template="plotly_white",
        title="Grey Wolf Optimizer on the Rosenbrock Function",
        autosize=False,
        width=w,
        height=h,
        margin=dict(l=10, r=10, b=50, t=90),
        legend=dict(
            x=xLeg,
            y=yLeg,
            traceorder="normal",
            font=dict(
                size=16,
            ),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(frameDuration)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.25,
                "y": 0.4,
            }
        ],
        # sliders = sliders
    )

    if notebook:
        pyo.iplot(fig, image_width=figSize[0], image_height=figSize[1])
    else:
        fig.show()
    return


def createFrames(popTensor, bestTensor):

    frameList = []
    off = 1.5

    for t in range(len(popTensor)):

        # Population
        xPop, yPop, zPop = popTensor[t][0], popTensor[t][1], popTensor[t][2]

        # Best agents
        Alpha = bestTensor[t][:, 0]
        zAlpha = rosenbrockFunc(Alpha)
        Beta = bestTensor[t][:, 1]
        zBeta = rosenbrockFunc(Beta)
        Delta = bestTensor[t][:, 2]
        zDelta = rosenbrockFunc(Delta)

        frameList.append(
            go.Frame(
                data=[
                    go.Scatter3d(  # Population
                        x=xPop,
                        y=yPop,
                        z=zPop + off,
                        mode="markers",
                        marker=dict(symbol="circle", color="rgb(12, 200, 17)", size=4),
                        name="Lobos caçadores",
                    ),
                    go.Scatter3d(  # Alpha Wolf
                        x=[Alpha[0]],
                        y=[Alpha[1]],
                        z=[zAlpha + off],
                        mode="markers",
                        marker=dict(symbol="diamond", color="rgb(12, 13, 130)", size=4),
                        name="Lobo Alfa",
                    ),
                    go.Scatter3d(
                        x=[Beta[0]],
                        y=[Beta[1]],
                        z=[zBeta + off],
                        mode="markers",
                        marker=dict(
                            symbol="diamond", color="rgb(220, 221, 28)", size=4
                        ),
                        name="Lobo Beta",
                    ),
                    go.Scatter3d(
                        x=[Delta[0]],
                        y=[Delta[1]],
                        z=[zDelta + off],
                        mode="markers",
                        marker=dict(symbol="diamond", color="rgb(255, 6, 0)", size=4),
                        name="Lobo Delta",
                    ),
                ],
                traces=[1, 2, 3, 4],
            )
        )

    return frameList


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }
