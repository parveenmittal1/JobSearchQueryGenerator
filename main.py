import spacy
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import argparse
import random
import numpy as np
import logging

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Initialize a BERT pipeline for semantic similarity
bert_pipeline = pipeline("feature-extraction", model="bert-base-uncased")

# Load pre-trained GPT-2 model and tokenizer for query generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token
model = GPT2LMHeadModel.from_pretrained("gpt2")


def calculate_bert_similarity(text1, text2):
    """Calculate cosine similarity using BERT embeddings."""
    embeddings1 = bert_pipeline(text1)[0][0]
    embeddings2 = bert_pipeline(text2)[0][0]

    dot_product = sum(a * b for a, b in zip(embeddings1, embeddings2))
    magnitude1 = sum(a ** 2 for a in embeddings1) ** 0.5
    magnitude2 = sum(b ** 2 for b in embeddings2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0


def prioritize_technologies(job_title, tech_hierarchy):
    """Select the best matching technology for the job title."""
    best_tech = None
    max_similarity = -1

    # Using the tech_hierarchy to identify the best tech match
    for categories in tech_hierarchy.values():
        for tech_info in categories.values():
            for tech_name in tech_info.keys():
                similarity = calculate_bert_similarity(job_title, tech_name)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_tech = tech_name

    return best_tech


def select_weighted_sites(priority_sites, max_sites=2):
    """Select a diverse set of sites based on priority and randomness."""
    sites = list(priority_sites.keys())
    scores = np.array(list(priority_sites.values()))
    probabilities = scores / scores.sum()

    # Get the number of high-priority and low-priority sites to be selected
    high_priority_count = min(1, max_sites)  # At least 2 high-priority sites
    low_priority_count = max_sites - high_priority_count

    # Select high-priority sites
    high_priority_sites = np.random.choice(
        [site for site in sites if priority_sites[site] >= 1],
        size=high_priority_count,
        replace=False
    )

    # Select low-priority sites
    if low_priority_count > 0:
        low_priority_sites = np.random.choice(
            [site for site in sites if priority_sites[site] < 4],
            size=min(low_priority_count, len(sites)),
            replace=False
        )
        selected_sites = np.concatenate((high_priority_sites, low_priority_sites))
    else:
        selected_sites = high_priority_sites

    return selected_sites.tolist()


def construct_query(sites, job_title, selected_tech, employment_type, location=None):
    """Construct a job search query."""
    site_query = ' OR '.join([f'site:{site}' for site in sites])
    query_parts = [site_query, f'"{job_title}"', f'"{selected_tech}"', f'"{employment_type}"']
    if location:
        query_parts.append(f'"{location}"')

    return ' '.join(query_parts)


def generate_queries(priority_sites, job_titles, tech_hierarchy, employment_types, location, output_file):
    """Generate job search queries for various job sites and save them to a file."""
    sorted_job_titles = sorted(job_titles.items(), key=lambda item: item[1], reverse=True)
    count = 0
    with open(output_file, "w") as file:  # Open output file for writing queries
        for job_title, _ in sorted_job_titles:
            selected_tech = prioritize_technologies(job_title, tech_hierarchy)
            if not selected_tech:
                continue

            selected_sites = select_weighted_sites(priority_sites)

            for employment_type in employment_types:
                constructed_query = construct_query(selected_sites, job_title, selected_tech, employment_type, location)
                # Write the constructed query to the file
                print(count)
                count = count+1
                file.write(f"{constructed_query}\n")  # Output required formatted query


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate job search queries for data engineering roles in Azure and AWS.")
    parser.add_argument('--location', type=str, default='Remote', help="Location for job search")
    parser.add_argument('--output', type=str, default='queries.txt', help="Output file for saving queries")

    args = parser.parse_args()

    # Define prioritized sites and their associated scores
    priority_sites = {
        # High priority job portals
        "linkedin.com": 5,
        "dice.com": 5,
        "indeed.com": 5,
        "monster.com": 5,
        "glassdoor.com": 5,
        "brillio.com": 5,
        "ziprecruiter.com": 5,
        "simplyhired.com": 5,

        # Mid-level job portals
        "tcs.com": 4,
        "roberthalf.com": 4,
        "teksystems.com": 4,
        "apexsystems.com": 4,
        "infosys.com": 4,
        "nagarro.com": 4,
        "capgemini.com": 4,
        "hays.com": 4,
        "randstad.com": 4,

        # Other reputable sites
        "collabera.com": 3,
        "beaconhillstaffing.com": 3,
        "insightglobal.com": 3,
        "kforce.com": 3,
        "mckinsey.com": 3,
        "hexaware.com": 3,
        "vaco.com": 3,
        "expresspros.com": 3,
        "manpower.com": 3,
        "aerotek.com": 3,

        # Industry-specific platforms
        "hired.com": 3,
        "angel.co": 3,
        "flexjobs.com": 3,
        "remote.co": 3,
        "jobs.github.com": 3,
        # ...additional job portals...
    }

    # Define job titles with their priority scores
    job_titles = {
        "Data Engineer": 5,
        "Azure Data Engineer": 5,
        "AWS Data Engineer": 5,
        "Data Modeler": 4,
        "ETL Developer": 4,
        "ELT Developer": 4,
        "Data Analytics Engineer": 4,
        "Databricks Engineer": 5,
        "Databricks Developer": 5,
        "Snowflake Data Engineer": 5,
        "Snowflake Engineer": 5,
        "Snowflake Developer": 5,
        "Data Architect": 5,
        "Big Data Engineer": 5,
        "Data Warehouse Engineer": 4,
        "Data Integration Engineer": 4,
        "Data Pipeline Engineer": 4,
        "DataOps Engineer": 4,
        "Analytics Engineer": 4,
    }
    # Technology hierarchy focusing on Azure and AWS for Data Engineering
    tech_hierarchy = {
        "Cloud Platforms": {
            "Azure": {"Azure Data Factory": 5, "ADF": 5},
            "AWS": {"AWS Glue": 5},
            "Other": {"Databricks": 5, "Snowflake": 4, "Apache Airflow": 3}
        },
        "Data Processing": {
            "Spark": {"Apache Spark": 5, "PySpark": 5},
            "Other": {"SQL": 5, "Python": 5, "Hadoop": 4, "Kafka": 3}
        },
        "Data Modeling": {
            "ETL": {"DBT": 2},
            "Data Warehousing": {"Redshift": 3},
            "Infrastructure": {"Terraform": 4}
        },
        "Visualization": {
            "Microsoft": {"Power BI": 2},
            "Tableau": {"Tableau": 2}
        },
        "Orchestration": {
            "Kubernetes": {"Kubernetes": 2},
            "Docker": {"Docker": 3}
        },
        "Legacy": {
            "SSIS": {"SSIS": 0.5},
            "Big Data": {"Big Data": 3}
        }
    }
    employment_types = ["Full-time", "Part-time", "OPT"]

    # Generate queries based on the provided input
    generate_queries(
        priority_sites,
        job_titles,
        tech_hierarchy,
        employment_types,
        args.location,
        args.output,
    )


if __name__ == "__main__":
    main()
