import tkinter
import customtkinter

# Modes: system (default), light, dark
customtkinter.set_appearance_mode("System")
# Themes: blue (default), dark-blue, green
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("640x480")


def button_function():
    print("button pressed")


def combobox_callback(choice):
    print("combobox dropdown clicked:", choice)


combobox_var = customtkinter.StringVar(value="option 2")
combobox = customtkinter.CTkComboBox(app, values=["option 1", "option 2"],
                                     command=combobox_callback, variable=combobox_var)
combobox_var.set("option 2")
combobox.place(relx=0.75, rely=0.5, anchor=tkinter.CENTER)

entry = customtkinter.CTkEntry(app, placeholder_text="CTkEntry")
entry.place(relx=0.25, rely=0.25, anchor=tkinter.CENTER)

# Use CTkButton instead of tkinter Button
button = customtkinter.CTkButton(
    master=app, text="CTkButton", command=button_function)
button.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

app.mainloop()
