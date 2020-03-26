# Escape Room

Agents use reinforcement learning to learn
how to navigate and escape a small room as
quickly as possible.

![greetings](/screenshots/agent.png)

This is the final project for the Reinforcement Learning Practical course from the University of Groningen.

by s3301419 & s3324818

email s3301419@student.rug.nl

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

![room1 GUI](/screenshots/room1.png)
   
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

![room2 GUI](/screenshots/room2.png)

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

![Q-table visualizer](/screenshots/q-visualizer.png)

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