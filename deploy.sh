#!/bin/bash

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null
then
    echo "Heroku CLI is not installed. Please install it first."
    echo "Visit: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    echo "Visit: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if logged in to Heroku
heroku_logged_in=$(heroku auth:whoami 2>&1 | grep -c "not logged in")
if [ $heroku_logged_in -eq 1 ]
then
    echo "You are not logged in to Heroku. Please login first."
    heroku login
fi

# Prompt for app name
read -p "Enter a name for your Heroku app: " app_name

# Create Heroku app
echo "Creating Heroku app: $app_name"
heroku create $app_name

# Set stack to container
echo "Setting Heroku stack to container"
heroku stack:set container -a $app_name

# Check if git is initialized
if [ ! -d .git ]
then
    echo "Initializing git repository"
    git init
    git add .
    git commit -m "Initial commit for Heroku deployment"
fi

# Add Heroku remote
echo "Adding Heroku remote"
heroku git:remote -a $app_name

# Push to Heroku
echo "Deploying to Heroku. This may take a few minutes..."
git push heroku master

# Open the app
echo "Deployment complete. Opening app..."
heroku open -a $app_name

echo "Your OCT Image Analysis app is now deployed at: https://$app_name.herokuapp.com"
echo "Note: If you encounter issues, check Heroku logs with: 'heroku logs --tail'"