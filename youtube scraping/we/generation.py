# Define the nested dictionary
import json

plant_parameters = {
    "general": {
        "inputs": ["plant_type", "plant_variety", "plant_health", "growth_stage"],
        "outputs": ["Day Temperature (°C)", "Night Temperature (°C)", "Optimal Humidity (%)", "Nitrogen (mg/L)",
                    "Phosphorus (mg/L)", "Potassium (mg/L)", "Calcium (mg/L)", "Magnesium (mg/L)", "Sulfur (mg/L)",
                    "Common Pests", "Common Diseases", "Plant Spacing (cm)", "Plant Density (Plants/m²)",
                    "Expected Yield Quantity (kg/m²)", "Harvesting Time", "Weekly Growth (cm/week)",
                    "Plant Height (cm)", "Leaf Color (RGB)"],
    },
    "Environment": {
        "Outdoor": {
            "inputs": ["climate"],
            "outputs": []
        },
        "Indoor": {
            "inputs": ["light_type"],
            "outputs": ["Light Type", "Light Spectrum", "Light Duration (hours/Day)", "Light Intensity"]
        },
        "Shadehouse": {
            "inputs": ["shade_material"],
            "outputs": ["Shade Percentage (%)", "Light Type", "Light Spectrum", "Light Duration (hours/Day)",
                        "Light Intensity"]
        },
        "Greenhouse": {
            "inputs": ["heating_system", "cooling_system", "ventilation_system", "humidification_system",
                       "dehumidification_system"],
            "outputs": ["Heating Setpoint (°C)", "Cooling Setpoint (°C)", "Ventilation Rate",
                        "Humidification Setpoint (%)", "Dehumidification Setpoint (%)", "Light Type", "Light Spectrum",
                        "Light Duration (hours/Day)", "Light Intensity"]
        },
    },
    "Farming type": {
        "Soil-Based": {
            "inputs": ["soil_type", "irrigation_method"],
            "outputs": ["Soil pH", "Soil Moisture (%)", "Watering duration (Minutes)",
                        "Watering Frequency (Times per Day)", "Recommended Fertilizer",
                        "Fertilizer Application Rate (g/m²)", "Fertilizer Application Frequency (Times per week)"]
        },
        "Soilless": {
            "inputs": ["growing_medium", "irrigation_method"],
            "outputs": ["Growing medium pH", "Growing Medium Water content (%)", "Watering duration (Minutes)",
                        "Watering Frequency (Times per Day)", "Recommended Fertilizer",
                        "Fertilizer Application Rate (g/m²)", "Fertilizer Application Frequency (Times per week)"]
        },
        "Hydroponic": {
            "inputs": ["Hydroponic_system_type", "Growing_medium"],
            "outputs": ["Water pH", "EC Value (dS/m)", "Dissolved Oxygen (mg/L)", "Iron (mg/L)", "Manganese (mg/L)",
                        "Zinc (mg/L)", "Copper (mg/L)", "Boron (mg/L)", "Molybdenum (mg/L)", "Chlorine (mg/L)"]
        },
        "Aeroponic": {
            "inputs": ["Aeroponic_system_type", "Growing_medium"],
            "outputs": ["Water pH", "EC Value (dS/m)", "Mist Frequency (min)", "Mist Duration (sec)", "Iron (mg/L)",
                        "Manganese (mg/L)", "Zinc (mg/L)", "Copper (mg/L)", "Boron (mg/L)", "Molybdenum (mg/L)",
                        "Chlorine (mg/L)"]
        },
        "Aquaponic": {
            "inputs": ["Aquaponic_system_type", "Fish_Type"],
            "outputs": ["Water pH", "EC Value (dS/m)", "Dissolved Oxygen (mg/L)", "Fish Feed Type",
                        "Fish Feed Quentity (g/Day)", "Water Temperature (°C)", "Ammonia Level (ppm)",
                        "Nitrite Level (ppm)", "Nitrate Level (mg/L)", "Iron (mg/L)", "Manganese (mg/L)", "Zinc (mg/L)",
                        "Copper (mg/L)", "Boron (mg/L)", "Molybdenum (mg/L)", "Chlorine (mg/L)"]
        },
        "Vertical Farming": {
            "inputs": ["Vertical_Farming_system_type", "Growing_medium"],
            "outputs": ["Water pH", "EC Value (dS/m)", "Watering duration (Minutes)",
                        "Watering Frequency (Times per Day)", "Iron (mg/L)", "Manganese (mg/L)", "Zinc (mg/L)",
                        "Copper (mg/L)", "Boron (mg/L)", "Molybdenum (mg/L)", "Chlorine (mg/L)"]
        },
    }

}


def generate_prompt(plant_parameters, plant_type, plant_variety, plant_health, growth_stage, environment, farming_type,
                    additional_inputs):
    # Start with the general inputs
    inputs = {
        "Plant Type": plant_type,
        "Plant Variety": plant_variety,
        "Current Plant Health": plant_health,
        "Growth Stage": growth_stage
    }

    # Add environment-specific inputs
    if environment in plant_parameters["Environment"]:
        for env_input in plant_parameters["Environment"][environment]["inputs"]:
            inputs[env_input.replace("_", " ").title()] = additional_inputs.get(env_input, "Unknown")

    # Add farming-type-specific inputs
    if farming_type in plant_parameters["Farming type"]:
        for farm_input in plant_parameters["Farming type"][farming_type]["inputs"]:
            inputs[farm_input.replace("_", " ").title()] = additional_inputs.get(farm_input, "Unknown")

    # Create the introductory paragraph
    intro = "You are an expert in plant care and agriculture. Provide detailed and accurate plant care recommendations based on the given inputs. Use precise values, and ensure every parameter is addressed. If a value is unknown, state 'Unknown' or use a placeholder.\n"
    intro += f"Please provide detailed plant care recommendations for a {plant_type} plant of the {plant_variety} variety, currently in {plant_health} condition, at the {growth_stage} stage, grown {farming_type} in an {environment} environment."

    # Append additional details based on the environment
    if environment in plant_parameters["Environment"]:
        for env_input in plant_parameters["Environment"][environment]["inputs"]:
            intro += f" {env_input.replace('_', ' ').title()} is {inputs[env_input.replace('_', ' ').title()]}."

    intro += " Include optimal conditions for the following parameters, using single numerical values only, with the specified units. Keep a value concise if expressed with a word or simple term:\n"

    # List the outputs
    outputs = plant_parameters["general"]["outputs"]
    if environment in plant_parameters["Environment"]:
        outputs += plant_parameters["Environment"][environment]["outputs"]
    if farming_type in plant_parameters["Farming type"]:
        outputs += plant_parameters["Farming type"][farming_type]["outputs"]

    # List the required recommendations
    intro += "\n".join(outputs) + "\n\n"

    # Example JSON output format
    example_json = {
        "Plant Parameters": inputs,
        "Recommendations": {output: "Specify optimal value" for output in outputs}
    }

    # Assemble the final prompt
    prompt = intro
    prompt += "The response must be in JSON format, as shown in the example below. Avoid providing approximate values or placeholders unless the information is truly unavailable.\n\n"
    prompt += "Example JSON output:\n"
    prompt += json.dumps(example_json, indent=4)

    return prompt


# Example usage
additional_inputs = {
    "climate": "Mediterranean",
    "soil_type": "Loamy",
    "irrigation_method": "Drip"
}

prompt = generate_prompt(
    plant_parameters=plant_parameters,
    plant_type="Tomato",
    plant_variety="Cherry Tomato",
    plant_health="Healthy",
    growth_stage="Vegetative",
    environment="Outdoor",
    farming_type="Soil-Based",
    additional_inputs=additional_inputs
)

print(prompt)

#  Example Output
# You are an expert in plant care and agriculture. Provide detailed and accurate plant care recommendations based on the given inputs. Use precise values, and ensure every parameter is addressed. If a value is unknown, state 'Unknown' or use a placeholder.
# Please provide detailed plant care recommendations for a Tomato plant of the Cherry Tomato variety, currently in Healthy condition, at the Vegetative stage, grown Soil-Based in an Outdoor environment. Climate is Mediterranean. Include optimal conditions for the following parameters, using single numerical values only, with the specified units. Keep a value concise if expressed with a word or simple term:
# Day Temperature (°C)
# Night Temperature (°C)
# Optimal Humidity (%)
# Nitrogen (mg/L)
# Phosphorus (mg/L)
# Potassium (mg/L)
# Calcium (mg/L)
# Magnesium (mg/L)
# Sulfur (mg/L)
# Common Pests
# Common Diseases
# Plant Spacing (cm)
# Plant Density (Plants/m²)
# Expected Yield Quantity (kg/m²)
# Harvesting Time
# Weekly Growth (cm/week)
# Plant Height (cm)
# Leaf Color (RGB)
# Soil pH
# Soil Moisture (%)
# Watering duration (Minutes)
# Watering Frequency (Times per Day)
# Recommended Fertilizer
# Fertilizer Application Rate (g/m²)
# Fertilizer Application Frequency (Times per week)

# The response must be in JSON format, as shown in the example below. Avoid providing approximate values or placeholders unless the information is truly unavailable.

# Example JSON output:
# {
#     "Plant Parameters": {
#         "Plant Type": "Tomato",
#         "Plant Variety": "Cherry Tomato",
#         "Current Plant Health": "Healthy",
#         "Growth Stage": "Vegetative",
#         "Climate": "Mediterranean",
#         "Soil Type": "Loamy",
#         "Irrigation Method": "Drip"
#     },
#     "Recommendations": {
#         "Day Temperature (\u00b0C)": "Specify optimal value",
#         "Night Temperature (\u00b0C)": "Specify optimal value",
#         "Optimal Humidity (%)": "Specify optimal value",
#         "Nitrogen (mg/L)": "Specify optimal value",
#         "Phosphorus (mg/L)": "Specify optimal value",
#         "Potassium (mg/L)": "Specify optimal value",
#         "Calcium (mg/L)": "Specify optimal value",
#         "Magnesium (mg/L)": "Specify optimal value",
#         "Sulfur (mg/L)": "Specify optimal value",
#         "Common Pests": "Specify optimal value",
#         "Common Diseases": "Specify optimal value",
#         "Plant Spacing (cm)": "Specify optimal value",
#         "Plant Density (Plants/m\u00b2)": "Specify optimal value",
#         "Expected Yield Quantity (kg/m\u00b2)": "Specify optimal value",
#         "Harvesting Time": "Specify optimal value",
#         "Weekly Growth (cm/week)": "Specify optimal value",
#         "Plant Height (cm)": "Specify optimal value",
#         "Leaf Color (RGB)": "Specify optimal value",
#         "Soil pH": "Specify optimal value",
#         "Soil Moisture (%)": "Specify optimal value",
#         "Watering duration (Minutes)": "Specify optimal value",
#         "Watering Frequency (Times per Day)": "Specify optimal value",
#         "Recommended Fertilizer": "Specify optimal value",
#         "Fertilizer Application Rate (g/m\u00b2)": "Specify optimal value",
#         "Fertilizer Application Frequency (Times per week)": "Specify optimal value"
#     }
# }
