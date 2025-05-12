import csv
import os

def create_jira_csv():
    # Your email for assignments
    your_email = "alec.posner1@gmail.com"
    
    # -------------------------------------------------------------------------
    # Enhanced sprint_plan: Epics/stories incorporating advanced technical details
    # from the comprehensive analysis of mockup-to-code technologies
    # -------------------------------------------------------------------------
    sprint_plan = {
        "Epic 1: Foundational Research Implementation": [
            {
                "summary": "Implement CNN-LSTM architecture for visual feature extraction",
                "description": (
                    "Develop a system based on the pix2code architecture that uses Convolutional Neural Networks for "
                    "visual feature extraction from design mockups and Long Short-Term Memory Networks for sequence "
                    "modeling of code tokens. Include dual attention mechanisms to align visual and syntactic features.\n\n"
                    "Acceptance Criteria:\n"
                    "- CNN architecture implemented for visual feature extraction\n"
                    "- LSTM networks integrated for code token sequence modeling\n"
                    "- Dual attention mechanisms established for feature alignment\n"
                    "- End-to-end trainable architecture demonstrated\n"
                    "- References: pix2code framework (2017), arXiv:1705.07962"
                )
            },
            {
                "summary": "Create spatial relationship modeling system",
                "description": (
                    "Develop a grid-based element detection system that accurately models spatial relationships between UI "
                    "components. Optimize the system to handle complex layouts and nested elements.\n\n"
                    "Acceptance Criteria:\n"
                    "- Grid-based element detection implemented with >85% accuracy\n"
                    "- Spatial relationship modeling validated against human perception\n"
                    "- Inference speed optimization to <2s per sample (baseline: 6.8s)\n"
                    "- References: pix2code evaluation (2023), arXiv:2406.19898v3"
                )
            },
            {
                "summary": "Develop design linting system for UI anti-patterns",
                "description": (
                    "Create a comprehensive design linting system that can detect common UI anti-patterns including "
                    "fragmented visual elements, inconsistent spacing hierarchies, and color contrast violations. "
                    "Ensure WCAG 2.1 compliance checks are integrated.\n\n"
                    "Acceptance Criteria:\n"
                    "- Detection of fragmented visual elements with >85% accuracy\n"
                    "- Identification of inconsistent spacing hierarchies with >90% precision\n"
                    "- Color contrast validation for WCAG 2.1 compliance\n"
                    "- References: Prototype2Code framework (May 2024), arXiv:2405.04975"
                )
            },
            {
                "summary": "Build flexbox-centric layout engine",
                "description": (
                    "Implement a layout engine focused on flexbox that can generate responsive code supporting multiple "
                    "breakpoint configurations. Ensure the engine produces clean, maintainable CSS.\n\n"
                    "Acceptance Criteria:\n"
                    "- Support for at least 12 breakpoint configurations\n"
                    "- Generated code meets >4.5/5 code readability scores\n"
                    "- Layout accuracy >85% compared to original designs\n"
                    "- References: Prototype2Code benchmarks, NAACL 2025 (aclanthology.org/2025.naacl-long.199/)"
                )
            }
        ],
        "Epic 2: Perceptual Intelligence and Grouping": [
            {
                "summary": "Implement perceptual grouping algorithm",
                "description": (
                    "Develop an algorithm that clusters UI elements with high alignment to human designer groupings. "
                    "The system should identify related components and maintain design hierarchy in generated code.\n\n"
                    "Acceptance Criteria:\n"
                    "- Clustering accuracy >90% compared to human designer groupings\n"
                    "- Preservation of design hierarchy in generated code\n"
                    "- Integration with the layout engine for structural code generation\n"
                    "- References: Prototype2Code (2024), perceptual grouping algorithm with 94.3% alignment"
                )
            },
            {
                "summary": "Create visual fidelity assessment using CLIP embeddings",
                "description": (
                    "Implement a visual fidelity assessment system using CLIP-based similarity scoring to evaluate "
                    "how closely the rendered code matches the original design mockup.\n\n"
                    "Acceptance Criteria:\n"
                    "- CLIP-based similarity scoring system implemented (0-1 scale)\n"
                    "- Benchmark against existing solutions (target >0.85 similarity score)\n"
                    "- Integration with the evaluation pipeline\n"
                    "- References: Design2Code benchmark, SALT-NLP/Design2Code"
                )
            },
            {
                "summary": "Develop element matching and IoU calculation",
                "description": (
                    "Create a system to evaluate element matching between mockup and rendered code using bounding box "
                    "overlap (IoU) and text content accuracy metrics.\n\n"
                    "Acceptance Criteria:\n"
                    "- Bounding box detection and IoU calculation implemented\n"
                    "- Text content accuracy evaluation with >90% precision\n"
                    "- Visual reporting dashboard for element matching\n"
                    "- References: Design2Code multimodal assessment framework"
                )
            },
            {
                "summary": "Implement color reproduction and contrast optimization",
                "description": (
                    "Develop algorithms for accurate color reproduction and automatic contrast optimization to ensure "
                    "visual fidelity and accessibility compliance.\n\n"
                    "Acceptance Criteria:\n"
                    "- Color accuracy measurement using ΔE (target <5 for >75% cases)\n"
                    "- Automatic contrast adjustment for accessibility\n"
                    "- Integration with WCAG validation layers\n"
                    "- References: Gemini Pro Vision benchmark (ΔE 4.5 in 38.2% cases)"
                )
            }
        ],
        "Epic 3: Accessibility and Semantic HTML": [
            {
                "summary": "Implement WCAG validation layers",
                "description": (
                    "Develop post-generation validation checks to ensure compliance with Web Content Accessibility Guidelines. "
                    "Include automated testing and remediation suggestions.\n\n"
                    "Acceptance Criteria:\n"
                    "- WCAG 2.1 validation checks implemented\n"
                    "- Automated accessibility testing integrated\n"
                    "- Remediation suggestions for identified issues\n"
                    "- References: Design2Code benchmark, screen reader compatibility metrics"
                )
            },
            {
                "summary": "Create semantic HTML enforcement system",
                "description": (
                    "Implement a system to enforce proper semantic HTML element usage during code generation, ensuring "
                    "proper structure and accessibility.\n\n"
                    "Acceptance Criteria:\n"
                    "- Strict element typing rules implemented\n"
                    "- Semantic structure validation\n"
                    "- ARIA attributes automatically included where appropriate\n"
                    "- References: Leading commercial solutions, Design2Code benchmark"
                )
            },
            {
                "summary": "Develop contrast optimization algorithms",
                "description": (
                    "Create algorithms for automated color adjustment to optimize contrast ratios while maintaining "
                    "design aesthetics.\n\n"
                    "Acceptance Criteria:\n"
                    "- Color contrast analysis implemented\n"
                    "- Automatic adjustment algorithms for insufficient contrast\n"
                    "- Integration with the design linting system\n"
                    "- References: WCAG 2.1 contrast requirements, color reproduction benchmarks"
                )
            },
            {
                "summary": "Implement screen reader compatibility testing",
                "description": (
                    "Develop an automated testing system to evaluate screen reader compatibility of generated code, "
                    "aiming for >75% pass rate.\n\n"
                    "Acceptance Criteria:\n"
                    "- Automated screen reader testing implemented\n"
                    "- Compatibility issues identified and reported\n"
                    "- Remediation suggestions provided\n"
                    "- References: Design2Code screen reader compatibility (61.7% pass rate), benchmark targets"
                )
            }
        ],
        "Epic 4: Evaluation and Benchmarking": [
            {
                "summary": "Implement comprehensive evaluation dashboard",
                "description": (
                    "Create a dashboard to visualize and track all evaluation metrics including SSIM, layout accuracy, "
                    "code readability, and accessibility compliance.\n\n"
                    "Acceptance Criteria:\n"
                    "- Dashboard displaying all key metrics\n"
                    "- Historical tracking of improvements\n"
                    "- Comparison with benchmark results\n"
                    "- References: Design2Code benchmark, Prototype2Code metrics"
                )
            },
            {
                "summary": "Develop code readability assessment",
                "description": (
                    "Implement tools to evaluate generated code readability using established metrics and compare "
                    "with human-written code.\n\n"
                    "Acceptance Criteria:\n"
                    "- Code readability scoring implemented (5-point scale)\n"
                    "- Comparison with benchmark values (target >4.5/5)\n"
                    "- Integration with the evaluation dashboard\n"
                    "- References: Prototype2Code readability benchmarks (4.8/5)"
                )
            },
            {
                "summary": "Create human evaluation rubric",
                "description": (
                    "Develop a comprehensive rubric for human evaluation of generated code, including maintainability, "
                    "cross-browser compatibility, and accessibility compliance.\n\n"
                    "Acceptance Criteria:\n"
                    "- Evaluation rubric with 5-point Likert scales\n"
                    "- Training materials for evaluators\n"
                    "- Integration with the evaluation workflow\n"
                    "- References: Design2Code human evaluation methodology"
                )
            },
            {
                "summary": "Implement cross-browser compatibility testing",
                "description": (
                    "Develop automated testing for cross-browser compatibility of generated code across major browsers "
                    "and devices.\n\n"
                    "Acceptance Criteria:\n"
                    "- Testing across Chrome, Firefox, Safari, and Edge\n"
                    "- Mobile browser compatibility verification\n"
                    "- Visual comparison across platforms\n"
                    "- References: Design2Code cross-browser compatibility metrics"
                )
            }
        ],
        "Epic 5: Future Capabilities and Integration": [
            {
                "summary": "Develop real-time collaboration framework",
                "description": (
                    "Implement a framework for simultaneous design-code synchronization that allows real-time updates "
                    "between design tools and generated code.\n\n"
                    "Acceptance Criteria:\n"
                    "- Real-time synchronization between design and code\n"
                    "- Change tracking and conflict resolution\n"
                    "- API for design tool integration\n"
                    "- References: Future outlook in comprehensive analysis"
                )
            },
            {
                "summary": "Create self-healing system for UI drift correction",
                "description": (
                    "Develop a system that can automatically detect and correct UI drift between design intent and "
                    "rendered code.\n\n"
                    "Acceptance Criteria:\n"
                    "- UI drift detection algorithms implemented\n"
                    "- Automatic correction suggestions\n"
                    "- Integration with the evaluation pipeline\n"
                    "- References: Future outlook in comprehensive analysis"
                )
            },
            {
                "summary": "Implement full-stack generation capabilities",
                "description": (
                    "Extend the mockup-to-code system to generate not only frontend code but also integrated backend "
                    "services based on UI requirements.\n\n"
                    "Acceptance Criteria:\n"
                    "- Backend service code generation from UI mockups\n"
                    "- API endpoint creation based on UI components\n"
                    "- Data model inference from UI elements\n"
                    "- References: Future outlook in comprehensive analysis"
                )
            },
            {
                "summary": "Create IDE integration for seamless workflow",
                "description": (
                    "Develop plugins or extensions for popular IDEs to provide seamless integration of the mockup-to-code "
                    "system within existing development workflows.\n\n"
                    "Acceptance Criteria:\n"
                    "- Integration with VS Code, WebStorm, and other popular IDEs\n"
                    "- Context-aware code suggestions\n"
                    "- Design preview within the IDE\n"
                    "- References: Developer experience priorities in comprehensive analysis"
                )
            }
        ]
    }

    # Create CSV file
    csv_file = "nlp_mockup_coder.csv"
    
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header row
        writer.writerow([
            'Summary', 'Issue Type', 'Description', 'Priority',
            'Labels', 'Epic Link', 'Issue ID', 'Parent', 'Assignee', 'Reporter'
        ])
        
        epic_count = 0
        story_count = 0
        
        for epic, stories in sprint_plan.items():
            epic_count += 1
            epic_id = f"EPIC-{epic_count}"
            
            # Add the epic
            writer.writerow([
                epic,  # Summary
                'Epic',  
                (f"Implementation of state-of-the-art AI and NLP models for pixel-perfect mockup-to-code conversion "
                 f"based on 2025 benchmarks and research findings.\n\n{epic}"),  # Enhanced epic description
                'Medium',
                'NLP,MockupConverter,AI,Design2Code',
                '',  # Epic Link
                epic_id,
                '',  # Parent
                your_email,
                your_email
            ])
            
            # Add the stories for this epic
            for story in stories:
                story_count += 1
                story_id = f"STORY-{story_count}"
                
                writer.writerow([
                    story["summary"],
                    'Story',
                    story["description"],
                    'Medium',
                    'NLP,MockupConverter,Design2Code',  
                    epic,
                    story_id,
                    epic_id,
                    your_email,
                    your_email
                ])
    
    print(f"CSV file created: {os.path.abspath(csv_file)}")
    print(f"All tasks assigned to: {your_email}")
    print("\nInstructions for importing into Jira:")
    print("1. Log into your Jira Cloud instance")
    print("2. Choose > System")
    print("3. Under 'Import and Export', click 'External System Import'")
    print("4. Click 'CSV'")
    print("5. Upload your CSV file and follow the wizard")
    print("6. Map the Assignee and Reporter fields during the import")


if __name__ == "__main__":
    create_jira_csv()