
#!/bin/zsh

mlx_lm.generate --prompt "I want to make it so that whenever I spawn a terminal in a particular directory, the terminal searches for a .venv file. If it finds the .venv file, it activates the virtual machine. How do I do that?" --model "mlx-community/gemma-3-12b-it-4bit" --max-tokens 4000 --draft-model "mlx-community/gemma-3-270m-it-4bit"
