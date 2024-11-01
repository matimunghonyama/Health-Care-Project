# MediClass: South African Healthcare Service Classification Project

## Overview
MediClass is a comprehensive healthcare service classification system designed to automatically categorize and analyze healthcare interactions across South African public and private facilities. The project aims to improve healthcare resource allocation, identify service gaps, and better understand patient needs across different communities.

## Problem Statement
The healthcare sector processes thousands of daily patient interactions that need to be classified into standardized categories. This project develops a robust classification system to automatically process and categorize these healthcare services, supporting better healthcare delivery and resource management.

## Features
* Automated classification of healthcare services
* Document image processing and annotation
* Healthcare interaction analysis
* Service pattern recognition
* Quality metrics monitoring

## Technologies Used
* Python
* Dash (for interactive dashboard)
* Plotly
* Pandas
* Labelbox (for text annotation)
* CVAT (for image annotation)
* OpenCV (for pre-trained image annotation benchmarking)

## Project Components

### 1. Healthcare Service Categories
* Primary Care
* Specialist Care
* Emergency & Urgent Care
* Diagnostic Services
* Preventive Care
* Chronic Disease Management
* Mental Health Services
* Maternal & Child Health
* Rehabilitation Services
* Dental Services
* Pharmacy Services
* Alternative Medicine
* Administrative Services

### 2. Annotation System
#### Text Annotation (Labelbox)
* Entity Recognition
* Category Classification
* Pattern Recognition
* Special Case Identification

#### Image Annotation (CVAT)
* Semantic Segmentation
* Bounding Box Detection
* Keypoint Annotation
* Polygon Annotation
* 3D Cuboid Annotation

### 3. Interactive Dashboard
* Current Metrics Monitoring
* Trend Analysis
* Performance Tracking
* Quality Metrics Visualization

## Key Performance Indicators (KPIs)

### Annotation Accuracy
* Inter-annotator Agreement: >90% (Kappa score)
* Medical Terminology Accuracy: >98%
* Service Classification Accuracy: >95%
* Provider Identification Accuracy: >98%

### Data Completeness
* Required Fields: 100%
* Supporting Information: >95%
* Patient Demographics: >90%
* Geographic Coverage: All provinces

### Processing Standards
* Annotations per Hour: >40
* Quality Verification Rate: 100%
* Error Correction Time: <24 hours
* Documentation Completeness: 100%

## Installation

```bash
# Clone the repository
git clone https://github.com/matimunghonyama/Health-Care-Project.git

# Navigate to project directory
cd Health-Care-Project

# Install required dependencies
pip install -r requirements.txt

# Run the dashboard
python interactive_dashboard_text.py
```

## Usage

### Running the Dashboard
```python
python interactive_dashboard_text.py
```
The dashboard will be available at `http://localhost:8050`

### Annotation Guidelines
1. Text Annotation (Labelbox):
   * Follow the schema defined in `labelbox_schema.json`
   * Ensure all required fields are completed
   * Verify accuracy against provided standards

2. Image Annotation (CVAT):
   * Follow semantic segmentation guidelines
   * Implement nested hierarchy for bounding boxes
   * Maintain accuracy requirements for different elements

## Project Structure
```
Health-Care-Project/
│
├── interactive_dashboard_text.py    # Dashboard implementation
├── labelbox_schema.json            # Text annotation schema
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── data/                          # Data directory
└── models/                        # Model directory
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
[MIT License](LICENSE)

## Contact
Matimu Munghonyama - nghonyamamatimu@gmail.com

Project Link: [https://github.com/matimunghonyama/Health-Care-Project](https://github.com/matimunghonyama/Health-Care-Project)
