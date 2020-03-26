# Escape Room

![](/screenshots/agent.png)

Agents use reinforcement learning to learn
how to navigate and escape a small room as
quickly as possible.

This is the final project for the Reinforcement Learning Practical course from the University of Groningen.

by s3301419 & s3324818

email s3301419@student.rug.nl

## How it works

The _room_ is represented by a grid of N&times;M cells. Multiple _agents_ can inhabit this room, and try to escape the room by reaching the "goal" cell, while avoiding death. Each agent starts the escape with 2 point of health, and walking over broken glass lowers their health by 1 point. When their health drops to 0, they die. Each simulation round lasts for 200 time steps, after which the simulation is reset and the whole escape process starts again. Each agent gets 1 move per time-step. They can move in any of the 4 direction, or they can stand still. If two agents try to walk into the same tile, they "bump into each other" and both agents remain on the tile they started for that turn.

The agents use [Q-learning](https://en.wikipedia.org/wiki/Q-learning) to learn an optimal path through the room. All of the agents share the same single Q-table, so this is a shared Q-learning algorithm. The agents get rewarded for successfully escaping the room, and they get punished for dying. The agents are also punished for every time step in the simulation until they escape or die. This encourages the agents to escape the room as quickly as possible, as they don't want to accumulate a lot of punishment for just standing around. We use the [&epsilon;-greedy](https://jamesmccaffrey.wordpress.com/2017/11/30/the-epsilon-greedy-algorithm/) policy selection, meaning that agents at random pick either the best action so far, or a completely random action each turn.

The [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) definitely rears its ugly head here, and for this assignment we were asked to limit our state space to no more than 1,000,000 states. As a result of this, the agents have a very limited view of the environment. They can only see 2 tiles away in each of the 4 directions around them. They also don't exactly see the tiles in their vision - they only know whether another agent occupies them, and if the tile is glass, a door, or a bandage they can see whether the glass was shattered, or door was opened, or bandage taken. This is a huge handicap, especially since the agents cannot see diagonally. However, it does mean there are less than 1,000,000 states in a 8&times;9 room.

![](/screenshots/vision-radius.png)

This algorithm works decently well, the agents will generally discover a pretty good escape path through the maze in a relative small number of rounds. However, it is far from perfect. Q-learning is generally known to have flaws in dynamic, multi-agent environments such as this one. The agents don't appear to learn to coordinate very well - for example they won't move "in unison", or help each other escape. Since the agents don't see diagonally, they can't predict when another agent will bump into them around a corner, which can sometimes make them get stuck for long periods of time.

![](/screenshots/corner-case.png)

## The room

The consists of an N&times;M grid of cells, where both N and M are between 1 and 9. Each cell can be one of the following types:

#### Floor

This is the equivalent of empty space. Agents can freely inhabit these tiles.

#### Wall

![](/screenshots/wall-tile.png)

Agents cannot pass through these tiles by any means. As a side note, everything outside of the N&times;M room is also considered to be a wall.

#### Door

![](/screenshots/door-tile.png)

The door is a dynamic element in the room. The agents obviously cannot walk through them when they are closed. Rather, the first time an agent tries to move into a door cell, the door will be opened, but the agent will stay in place. The door will then stay open until the end of the simulation round since there is no way for the agents to close a door.

![](/screenshots/open-door-tile.png)

#### Glass

![](/screenshots/glass-tile.png)

Another dynamic element. Glass is very similar to doors - the agents first cannot walk through it. The first time an agent tried to move into a glass tile, the glass will shatter, and the agent will stay in place. If an agent then walks over the glass, they will get hurt and lose 1 health point. This is the only way agents can get hurt in the room.

![](/screenshots/broken-glass-tile.png)

#### Bandage

![](/screenshots/bandage-tile.png)

The bandage will heal any agent that walks over it back to the full 2 health points, and it will be consumed in the process. The bandage will be consumed even if the agent is _already_ at full health.

#### Goal

![](/screenshots/goal-tile.png)

This marks the exit of the room - when an agents walks into a goal tile, they are removed from the room. There can be multiple goals in the room.

## Options

There are two versions of the program. We
have worked very hard to create a GUI that
can visualize the enviromnent in real time
as the agents are learning. Some features
of the GUI like the Q-table visualizer have
significantly helped us understand what is
going on in the environment, why the agents
behave the way they do. For this reason we
recommend that you compile and run the GUI
version for the "full experience". However
we did use an external library to implement
the GUI, so compiling this version is a bit
more complicated. Because of this, there is
also an option to disable the GUI and avoid
the dependancy. Library files are provided
for you, in the same directory so we hope
that compiling the GUI won't be a problem.

## How to compiler with the GUI?

#### With GCC

```bash
$ gcc escape.c -std=c99 -L. -lm -lglfw3
```

#### With MSVC

```bash
$ cl escape.c glfw3.lib msvcrt.lib user32.lib gdi32.lib shell32.lib
```

#### With clang

```bash
$ clang escape.c -std=c99 -L. -lm -lglfw3
```

![](/screenshots/room1.png)
   
The GLFW library is the only dependancy, and we provide versions for [windows](/libglfw3.lib), [linux](/libglfw3.so), and [mac](/libglfw3.a). If they don't work for whatever reason and you still
want the GUI, get glfw3-dev from your
package manager, or [download GLFW](https://www.glfw.org/download.html) 
and try again. Sorry.

If you give up on compiling, there are
pre-compiled programs for [windows](/bin/escape.exe) and [linux](/bin/escape.out)
in the [`/bin`](/bin) directory.

## How to compile without the GUI? (NOGUI)

```bash
$ gcc escape.c -std=c99 -D NOGUI -lm
```

```bash
$ cl escape.c -D NOGUI
```

You can add optimization flags if you want
the program to run faster.

## How to run?

Simply run the exectuable. A prompt/window will appear and typing "h"
will show you a short reference of commands
you can use.

## How to reproduce the paper results?

Type "reproduce" into the prompt. If you are using the GUI, pressing `X` will bring up the prompt.

MAKE SURE that [room1.txt](/room1.txt), [room2.txt](/room2.txt) and [room3.txt](/room3.txt) all exist **in the same directory**.

## How to use the GUI?

Using the command line program (NOGUI) is
fairly straightforward, but using the GUI
is a bit more challenging since we didn't
implement any text rendering.

You can press `H` at any point to print
out a help reference to your terminal.

At any point you can press `X` to access
the command prompt version of the program.
For example, if you want to open a room
from file `ROOM.txt`:
1. press `X`, the window will minimize
2. type `load ROOM.txt` into the prompt

## Edit mode

When you launch the GUI, you will be in
_EDIT_ mode by default. This means you can
edit the room using your mouse.
- Use the `mouse wheel` to _select_ something.
- `Click` to _place_ your selection.
- `Right click` to _remove_ it.
- The top left of the screen will show
  your current selection.
- You can also _select_ something by `middle
  clicking` on it.
- If your current selection is an agent,
  you can `drag` the agents with your mouse.
- Use the `arrow keys` to _resize_ the room.
- You can press `CTRL+Z/Y` to _undo/redo_.
- You can also edit rooms by opening up
  a room text file with your text editor.
- When you exit the GUI your room will be
  saved as `room.txt`. This room will also
  be loaded when you launch the GUI.

## Run mode

When you are done editing the room, press
`ENTER` or `SPACE` to switch to _RUN_ mode.
In this mode the agents will start moving
around and trying to reach the goal.

![](/screenshots/room2.png)

- Use numbers `0`..`9` to control the speed
  of the simulation.
- After 200 turns, the simulation resets.
- Press `F` to run the simulation really
  quickly. Press `F` again to slow down.
- If you want to save the results to a file
  Press `X` and type `saveto YOURFILE` at
  the prompt. New results will now be added
  to `YOURFILE` as CSV files.
- You can _PAUSE_ by pressing `SPACE`. press
  `SPACE` again to _unpause_, or press `ENTER`
  to go back to _EDIT_ mode.

## Q-table visualizer

`V` toggles the Q-table _visualizer_. The
visualizer will will highlight the table
entries of agents as it updates.

![](/screenshots/q-visualizer.png)

- Each cell will split into 5 parts, 1 for
  each of the 5 actions an agent can take.
- Parts highlighted in _green_ have positive
  Q-values while parts highlighted
  in _red_ have nagative Q-values.
- Parts highlighted in _blue_ are the parts
  with the highest Q values, and are the
  ones that will probably be picked very
  often by the agents.
- When you enable the visualizer, a _green_
  outline will appear around the window.
  This indicates that you are visualizing
  Q-entries only for agents that have **2
  points of health** (which is the maximum).
- Press `V` again, and a _red_ outline will
  replace the _green_ one. You are now seeing
  the Q-entries only for agents that have
  1 point of health.

 have fun! :)