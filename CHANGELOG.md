# Change Log
---
## [Known Bug]
- In cases where two annotations overlap on the same slice, updating the nodule table results in the creation of two separate nodules. The expected behavior is to merge the two nodules in such scenarios.
- In specific cases, the program may not ask user to save the changes before exiting the program.

## [Unreleased]
### Fixed
- When editing the nodule more than once, there are some unexpected behaviors.
### Changed
- After selecting new patient, the program will not reset the model.
### Optimized
- Change loading model method to thread to avoid the program freezing.
- Removed redundant duplicate comments to optimize the code.

## [2.0.11] - 2023-10-28
### Added
- Add loading progress bar when progating the nodule.
### Fixed
- Fix the bug that it cannot segment multiple nodules on the same slice simultaneously.
- Removed redundant duplicate function calls to optimize the code.
## [2.0.10] - 2023-10-21
### Changed
- Hide the patient which is not inferenced by the model.
### Fixed
- Speed up the loading of the nodule table
- Set the time zone for displaying the last update time on the table to UTC+8 for consistency and accuracy in representation
- Set the time zone for saving log files to UTC+8 for consistency and accuracy in representation
- After loading the log, it cannot press "Update" button to update the nodule table.
## [2.0.9] - 2023-10-13
### Added
- Added User Option:
    - Implemented a user selection feature.
    - Users can now specify their identity by adding "--user" during program execution.
    - This feature assists in identifying the individual responsible for the final confirmation or decision-making process.
    ```shell
    python NoduleDetection_v14.py --user_name <user_name>
    ```
- Added a "Confirmed Status" feature.
    - Users can now view the current number of confirmations made by each user.
    - This information can be accessed by selecting the "Confirmed Status" option in the "Help" menu located at the top.
    - Users can now easily identify which user made a confirmation directly within the table located in the top left corner.
- Implemented scroll wheel functionality on the canvas.
    - Users can now conveniently switch between different slice by using the mouse scroll wheel to navigate upwards or downwards.
- Added a "Save" button in the right section.
    - The "Save" button allows users to save their editing history, providing the same functionality as pressing "Ctrl+S."
- Added automatic refresh feature
    - The table is automatically refreshed every 60 seconds by default. You can modify this by using help>Change Auto Refresh Frequency.
    - Added display of last refresh time in the top left area.
- Added "mark type" field in the nodule annotation table to display the type of the annotation. Currently supports two types: rectangle and polygon.
### Changed
- User Experience Improvement
    - Change the default mode of add new nodule button from "rectangle" to "polygon"
### Fixed
- Display Optimization:
    - Improved the functionality to view changes instantly for display labels and change point size without requiring a refresh by toggling.
- Display Label Bug Fix:
    - Fixed a bug in the display label section.
    - The bug was caused by missing the "group id" attribute for new labels, which has been addressed and resolved.
- Fix the unexpected behavior during annotation in hidden mode.
### Removed
- Code Cleanup
    Eliminated unnecessary code files:
    1. /libraries/selectDialog.py
    2. /libraries/text.py
    3. /libraries/vtk.py
## [2.0.8] - 2023-10-04
- First release