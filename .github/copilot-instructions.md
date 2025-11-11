Write code modularly, with clear separation of concerns.

Add docstrings and comments to explain the purpose of functions and complex code sections.
Add comments outlining the shapes of tensors at key points in the code to enhance readability and maintainability.
Try to minimize the number of lines of code written, while keeping clarity.

Use expressive variable and function names to improve code readability.

Remeber that the tdmpc2 conda env always needs to be activated when running code in this repo.

Never write fallbacks and default values. I want an error to occur if the code tries to access a missing config value.
In general I would rather have an error than running code that doesnt what it should