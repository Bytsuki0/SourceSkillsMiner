#!/bin/bash

USERS_FILE="github_users.txt"
MAIN_CONFIG="config_main.ini"
CONFIG_FILE="config.ini"
SCRIPT="ScoringSys.py"

# Check if the users file exists
if [[ ! -f "$USERS_FILE" ]]; then
    echo "Error: $USERS_FILE not found."
    exit 1
fi

# Check if config_main.ini exists
if [[ ! -f "$MAIN_CONFIG" ]]; then
    echo "Error: $MAIN_CONFIG not found."
    exit 1
fi

# Check if the scoring script exists
if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: $SCRIPT not found."
    exit 1
fi

# Read token from config_main.ini
TOKEN=$(grep -E "^\s*token\s*=" "$MAIN_CONFIG" | head -n1 | sed 's/.*=\s*//' | tr -d '[:space:]')

if [[ -z "$TOKEN" ]]; then
    echo "Error: Could not read token from $MAIN_CONFIG."
    exit 1
fi

echo "Token loaded from $MAIN_CONFIG."

# Read usernames line by line, skipping empty lines and comments
while IFS= read -r username || [[ -n "$username" ]]; do
    # Skip empty lines and lines starting with #
    [[ -z "$username" || "$username" == \#* ]] && continue

    echo "--------------------------------------------"
    echo "Processing user: $username"

    # Create config.ini for this user
    cat > "$CONFIG_FILE" <<EOF
[github]
username = $username
token = $TOKEN
EOF

    echo "config.ini created for: $username"

    # Run the scoring script and wait for it to finish
    python3 "$SCRIPT"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "Warning: $SCRIPT exited with code $EXIT_CODE for user: $username"
    else
        echo "Finished running $SCRIPT for user: $username"
    fi

done < "$USERS_FILE"

