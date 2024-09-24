# Connecting VSCode Jupyter to a compute node in HPC

## On KAYA

1. Create a slurm file to open a port on the compute node.

    File: `tunnel.slurm`

    ```bash
    #!/bin/bash -l
    #SBATCH --job-name=tunnel
    #SBATCH --partition=gpu
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=16G
    #SBATCH --gres=gpu:p100:1
    #SBATCH --export=ALL
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=21980614@student.uwa.edu.au
    #SBATCH --time=3-00:00:00

    # module load Anaconda3/2024.06 cuda/12.4

    module list

    nvcc --version

    nvidia-smi

    /usr/sbin/sshd -D -p 2222 -f /dev/null -h ${HOME}/.ssh/id_ecdsa
    ```

1. Add module load commands to `.bashrc` file as each Jupyter cell will run a new bash session so modules need to be loaded again and again.

    ```bash
    module load Anaconda3/2024.06 cuda/12.4
    ```

1. Check if KAYA has `id_ecdsa` file in `~/.ssh/` directory. If not, generate one using `ssh-keygen` command.

    ```bash
    ssh-keygen -t ecdsa -b 521 -f ${HOME}/.ssh/id_ecdsa
    ```

1. Add public key to the `~/.ssh/authorized_keys` file.

    ```bash
    cat ~/.ssh/id_ecdsa.pub >> ~/.ssh/authorized_keys
    ```

1. Change the permissions of `.ssh`.

    ```bash
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
    ```

1. Submit the slurm file.

    ```bash
    sbatch tunnel.sh
    ```

## On Local Machine

1. Open your ssh config file and add the following lines.

    File: `~/.ssh/config`

    ```bash
    Host login-node
      HostName kaya.hpc.uwa.edu.au
      User nchoong
      PreferredAuthentications publickey
      IdentityFile "~/.ssh/id_rsa"
      ServerAliveInterval 240
      ServerAliveCountMax 2

    Host compute-node
        HostName localhost
        Port 2222
        User nchoong
        IdentityFile "~/.ssh/id_rsa"
    ```

1. Check if you have `id_rsa` file in `~/.ssh/` directory. If not, generate one using `ssh-keygen` command.

    ```bash
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
    ```

1. Add public key to KAYA.

    ```bash
    ssh-copy-id -i ~/.ssh/id_rsa.pub nchoong@kaya.hpc.uwa.edu.au
    ```

1. Change the permissions of the `.ssh`.

    ```bash
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
    ```

1. Create a bash script to open a tunnel to the compute node.

    File: `connect_compute_node.sh`

    ```bash
    #!/bin/bash

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

    # Establish SSH tunnel
    ssh -L 2222:"$compute_node":2222 login-node

    # add -fN to the ssh command to run it in the background without shell
    # ssh -L 2222:"$compute_node":2222 login-node -fN
    ```

1. Change the permissions of the script.

    ```bash
    chmod +x connect_compute_node.sh
    ```

1. Or you can create a function in your `.bashrc` file.

    ```bash
    function connect_compute_node() {
      # Kill any existing SSH tunnel
      lsof -ti:2222 | xargs kill -9
      ps aux | grep "ssh -L 2222" | grep -v grep | awk '{print $2}' | xargs kill

      # Get the compute node name dynamically
      compute_node=$(ssh login-node squeue --me --name=tunnel --states=R -h -O NodeList | xargs)

      # Check if the compute node was found
      if [ -z "$compute_node" ]; then
        echo "No compute node found for the current job."
        return 1
      fi

      echo "Connecting to compute node: $compute_node"

      # Establish SSH tunnel
      ssh -L 2222:"$compute_node":2222 login-node
    }
    ```

1. Run the script or function.

    ```bash
    ./connect_compute_node.sh
    ```

    or

    ```bash
    connect_compute_node
    ```

1. Then open VSCode or another terminal and run the following command to connect to the compute node.

    ```bash
    ssh compute-node
    ```

    or

    ```bash
    ssh -p 2222 nchoong@localhost
    ```
