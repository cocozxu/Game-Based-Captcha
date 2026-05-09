# Trace-the-Tunnel

A browser game that records mouse trajectories (drag from the green dot to the red dot through a curved tunnel) and a pipeline for classifying human vs. AI agent traces.

## Run the Game

```bash
./setup.sh                          # one-time: create .venv, install deps
source .venv/bin/activate
python server.py                    # default mode — saves by client-supplied 'source'
python server.py --experiment agent_minimal   # experiment mode — forces all saves to data/agent_minimal/
```

The server runs at `http://localhost:5050`. In default mode, open it in a browser and play — sessions auto-save to `data/human/` on completion or failure.

**Experiment mode** locks the output folder for a named run: the server overwrites the `source` field on every save and refuses `source='human'`, so agent trajectories can't accidentally pollute the human dataset. Use it whenever you're collecting agent data (see below).

## Collecting Agent Data with Claude Code

**Important:** run Claude Code from a *separate, empty directory* — never from inside this repo. Claude Code behaves quite differently if it has access to the source code of the game. It had the tendency to stay in the middle somehow.

Setup:

```bash
mkdir ~/trace-the-tunnel-exp && cd ~/trace-the-tunnel-exp    # outside this repo
claude                                          # start Claude Code here
```

Give the agent only the URL (`http://localhost:5050`) and the task prompt from `experiments/prompts/`. Do not share file paths into this repo.

Then from this repo, in another terminal, start the server in experiment mode and launch the run:

```bash
source .venv/bin/activate && python server.py --experiment agent_minimal
./experiments/run.sh agent_minimal --rounds 3
```
