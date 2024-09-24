#!/bin/bash

# Make the script executable
# chmod +x connect_compute_node.sh

# Kill any existing SSH tunnel
lsof -ti:2222 | xargs kill -9
ps aux | grep "ssh -L 2222" | grep -v grep | awk '{print $2}' | xargs kill

# Get the compute node name dynamically
compute_node=$(ssh login-node squeue --me --name=tunnel --states=R -h -O NodeList | xargs)

# Check if the compute node was found
if [ -z "$compute_node" ]; then
  echo "No compute node found for the current job."
  exit 1
fi

echo "Connecting to compute node: $compute_node"

# Establish SSH tunnel in the background without a shell
ssh -L 2222:"$compute_node":2222 login-node

# Connect to the compute node through the tunnel
# ssh -p 2222 nchoong@localhost
