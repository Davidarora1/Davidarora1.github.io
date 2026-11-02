#!/usr/bin/env bash
# Downloads the public-domain archive clips (US National Archives via archive.org) into ./footage
set -e
cd "$(dirname "$0")"; mkdir -p footage; cd footage
get() { [ -s "$1" ] || curl -sSL -A 'Mozilla/5.0' -o "$1" "https://archive.org/download/$2"; }
get naga_burst.mp4  "428Npc2586NagasakiBurst/428-npc-2586%20Nagasaki%20Burst.mp4"
get hiro_damage.mp4 "USAF-11075/USAF-11075.mp4"
get hiro_life.mp4   "USAF-11079/USAF-11079.mp4"
get b29.mp4         "CS-3338/CS-3338.mp4"
get naga_damage.ogv "NPC-3243/428-npc-3243%20Atomic%20Bomb%20Damage%20Nagasaki%20Japan.ogv"
get train.mp4       "passenger_train/passenger_train_512kb.mp4"
get hiro_cloud.jpg  "propix.493984294/wars-wwii-japan-bombing-hiroshima-atomic-bomb_493984294_o.jpg"
echo "footage ready"
