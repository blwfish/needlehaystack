# needlestack — User Guide

Find photos by describing what's in them.
Type "steam locomotive pulling freight through mountain pass" and needlestack
finds your photos that match — no folders to browse, no filenames to remember.

**Your photos never leave your computer.** needlestack runs entirely on your
Mac — no account, no internet connection needed after setup, no uploads,
nothing sent anywhere.

Here's what's actually happening: when you index a folder, needlestack reads
each photo once and uses a locally-running AI to write a short description of
what's in it — something like *"a red caboose on a siding, pine trees in the
background, late afternoon light."* That description gets stored in a small
database on your Mac. The photos themselves are not copied or moved.

When you search, your words are passed to the same local AI, which quietly
expands them into related terms before searching the stored descriptions —
so "caboose" also matches "cabin car" and "waycar." The results you see are
drawn entirely from that local database; needlestack does not look at your
actual photo files during a search.

The AI is an open-source model called LLaVA, running inside
[Ollama](https://ollama.com) on your Mac. It has no network access and no
connection to any external service. If you want to verify that, you can
disconnect from the internet after installation and everything will still work.

---

## What you need

- A Mac running macOS Monterey (12) or later
- [Ollama](https://ollama.com) installed (free, downloads separately)
- About 14 GB of free disk space for setup
- Your photos in a folder somewhere on your Mac

---

## Installation

1. Download needlestack from GitHub — click the green **Code** button, then
   **Download ZIP**
2. Unzip it — your Mac will do this automatically if you double-click the file
3. A folder will appear in your Downloads (its name won't matter for long —
   see the next step). Move it to your home folder or Documents — somewhere
   you won't accidentally delete it — and rename it to **needlestack**.
   **Do not move or rename it again after installing.**
4. Open the **needlestack** folder and double-click **install-mac.sh**
5. If macOS says it can't open the file because it's from an unidentified
   developer, try right-clicking it and choosing **Open**, then click **Open**
   again in the dialog that appears. If macOS has already blocked it, open
   **System Settings → Privacy & Security**, scroll to the bottom, and click
   **Open Anyway** next to the needlestack entry.
6. A Terminal window will open and walk you through the rest

The installer will download the AI model (about 8 GB) — this is the part that
takes the longest. You can leave it running and come back.

---

## Indexing your photos

Indexing is a one-time process that reads through your photos and learns what's
in each one. You only need to do this once, though you can re-run it later if
you add more photos.

Open Terminal, then type (replacing the path with your actual photos folder):

```
cd ~/needlestack
pixi run needlestack index ~/Pictures/MyPhotos
```

A progress bar will show how far along it is. For a few thousand photos,
expect **one to three hours** — a good overnight job.

If it's interrupted for any reason (computer goes to sleep, you close Terminal),
just run the same command again. It picks up where it left off.

---

## Searching

Open **Needlestack** from Launchpad, or search for it with Spotlight
(⌘-Space, type "Needlestack"). The first time, macOS will warn you because
it's not signed — right-click it and choose **Open**, then click **Open**
in the dialog. If macOS has already blocked it, open
**System Settings → Privacy & Security**, scroll to the bottom, and click
**Open Anyway**.

A browser window will open automatically. Type what you're looking for and
press Enter or click Search.

**Tips for good searches:**

- Be specific: *"red brick station with covered platform"* works better than
  *"station"*
- Describe what you see: *"steam locomotive with large driving wheels"* rather
  than a model number
- Combine features: *"wooden freight car with reporting marks on the side"*
- Era helps: *"1950s diesel locomotive"*

Results are shown best-match first. Hover over a photo to see the description
needlestack generated for it. Click a photo to open it in Finder.

---

## Troubleshooting

### "Ollama is not installed"

needlestack needs Ollama to understand your photos. Go to
[ollama.com](https://ollama.com), download the Mac app, open it, and run the
installer again.

### "Ollama did not start"

Ollama may not have finished launching. Open the Ollama app from your
Applications folder, wait a minute, then try again.

### The browser opens but search returns nothing

This usually means indexing hasn't been run yet, or it didn't finish.
Check that you ran the index command and that it completed successfully.

### Search results don't look right

The quality of results depends on how well needlestack described your photos
during indexing. If results seem off, try:

- Using different words — *"waycar"* or *"cabin car"* instead of *"caboose"*
- Being more specific — add colours, materials, or era
- Checking a few photo descriptions by hovering over results — if the
  descriptions seem generic ("a train on some tracks"), the index may need
  to be rebuilt with a newer version of needlestack

### It was working and now it's not

Make sure Ollama is running — look for the Ollama icon in your menu bar
(top right of your screen). If it's not there, open Ollama from Applications.

### Something went wrong during installation

Run this command in Terminal to generate a diagnostic report:

```
cd ~/needlestack
pixi run needlestack doctor --out report.txt
```

This creates a file called **report.txt** in the needlestack folder.
Send that file to whoever gave you needlestack and they can figure out
what went wrong.

### Search is very slow

The first search after starting needlestack takes longer while the AI warms up —
usually 10–20 seconds. After that, searches should take about 5 seconds.
If every search is slow, your Mac may be under heavy load from other
applications.

### I added new photos — how do I index them?

Run the same index command you used the first time. needlestack will skip
photos it has already seen and only process the new ones.

---

## Getting help

If something isn't working and you can't figure it out from the above, run:

```
cd ~/needlestack
pixi run needlestack doctor --out report.txt
```

Send the resulting **report.txt** to whoever gave you needlestack.
It contains everything needed to diagnose the problem remotely —
no need to describe what you were doing or what the screen said.

---

## Uninstalling

1. Delete the **needlestack** folder
2. Delete **~/.needlestack** (this is the photo index — hidden folder in your
   home directory, find it in Terminal with `rm -rf ~/.needlestack`)
3. Ollama can be uninstalled separately from your Applications folder if you
   no longer need it
