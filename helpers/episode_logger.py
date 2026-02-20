import os
import csv
import time
import keyboard  # You might need to install this package using `pip install keyboard`

def create_new_file():
    file_name = input("Enter the name of the new Excel file (e.g., 'episodes.csv'): ")
    while os.path.exists(file_name):
        print(f"The file {file_name} already exists. Please choose a different name.")
        file_name = input("Enter the name of the new Excel file (e.g., 'episodes.csv'): ")
    return file_name

def continue_existing_file():
    file_name = input("Enter the name of the existing Excel file (e.g., 'episodes.csv'): ")
    while not os.path.exists(file_name):
        print(f"The file {file_name} does not exist. Please provide a valid file name.")
        file_name = input("Enter the name of the existing Excel file (e.g., 'episodes.csv'): ")
    return file_name

def get_last_episode_number(file_name):
    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if rows:
            last_episode = int(rows[-1][0].split('_')[1])
            return last_episode
    return 0

def main():
    print("Welcome to the Episode Notation Logger!")
    choice = input("Do you want to create a new Excel file (N) or continue an existing one (C)? (N/C): ").strip().lower()
    
    if choice == 'n':
        file_name = create_new_file()
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode_Number', 'Notation'])  # Write header
    elif choice == 'c':
        file_name = continue_existing_file()
        last_episode = get_last_episode_number(file_name)
    else:
        print("Invalid choice. Exiting.")
        return

    episode_number = last_episode + 1 if choice == 'c' else 1  # Start from the last episode or 1

    try:
        while True:
            # Print current episode number and wait for input
            print(f"Episode_{episode_number}: ", end="")
            notation = input()

            if notation == '':
                episode_number += 1
                continue  # Skip if no notation is provided
            
            # If 'Esc' is pressed, stop the script and save the file
            if keyboard.is_pressed('esc'):
                print(f"Saving data to {file_name}...")
                break

            # Write the episode number and notation to the CSV file
            with open(file_name, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"episode_{episode_number:03d}", notation])
            
            #print(f"Episode_{episode_number}: {notation}")  # Print the entry

            episode_number += 1  # Increment episode number
            time.sleep(0.2)  # Small delay to avoid rapid keypress issues

    except KeyboardInterrupt:
        print("\nProcess interrupted.")
    finally:
        print(f"Data has been saved to {file_name}")

if __name__ == "__main__":
    main()
