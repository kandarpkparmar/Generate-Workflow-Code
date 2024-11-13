from workflow_generator import GemmaWorkflowGenerator

generator = GemmaWorkflowGenerator()

# Example input
human_input = "Create a purchase order for purchasing of 500 computers and 1000 smartphones."
conversation_history = [
        {
            "type": "USER_INPUT",
            "content": human_input,
            "str_created_at": "2024-10-05 11:09:44.788438"
        }
    ]

output = generator.generate_workflow(human_input, conversation_history)
print(output)