# Azure Interview Preparation Guide

> **Focus Areas:** Azure Storage · Azure App Service · Azure AI Foundry · Azure AI Services

This guide provides end-to-end tutorials with visual architecture diagrams and curated interview questions to help you crack an Azure-focused interview.

---

## Table of Contents

1. [Azure Storage](#1-azure-storage)
   - [Overview & Architecture](#11-overview--architecture)
   - [End-to-End Tutorial](#12-end-to-end-tutorial)
   - [Interview Questions](#13-interview-questions)
2. [Azure App Service](#2-azure-app-service)
   - [Overview & Architecture](#21-overview--architecture)
   - [End-to-End Tutorial](#22-end-to-end-tutorial)
   - [Interview Questions](#23-interview-questions)
3. [Azure AI Foundry](#3-azure-ai-foundry)
   - [Overview & Architecture](#31-overview--architecture)
   - [End-to-End Tutorial](#32-end-to-end-tutorial)
   - [Interview Questions](#33-interview-questions)
4. [Azure AI Services](#4-azure-ai-services)
   - [Overview & Architecture](#41-overview--architecture)
   - [End-to-End Tutorial](#42-end-to-end-tutorial)
   - [Interview Questions](#43-interview-questions)

---

## 1. Azure Storage

### 1.1 Overview & Architecture

Azure Storage is Microsoft's cloud storage solution for modern data storage scenarios. It is durable, highly available, massively scalable, and accessible from anywhere in the world.

**Four Core Storage Services:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      AZURE STORAGE ACCOUNT                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌────────┐ │
│  │ BLOB STORAGE │  │ FILE STORAGE │  │  QUEUE   │  │ TABLE  │ │
│  │              │  │              │  │ STORAGE  │  │STORAGE │ │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │          │  │        │ │
│  │ │ Container│ │  │ │  Share   │ │  │ ┌──────┐ │  │┌──────┐│ │
│  │ │ ┌──────┐ │ │  │ │ ┌──────┐ │ │  │ │Msg 1│ │  ││Entity││ │
│  │ │ │Blob 1│ │ │  │ │ │File 1│ │ │  │ │Msg 2│ │  ││Entity││ │
│  │ │ │Blob 2│ │ │  │ │ │File 2│ │ │  │ │Msg 3│ │  ││Entity││ │
│  │ │ └──────┘ │ │  │ │ └──────┘ │ │  │ └──────┘ │  │└──────┘│ │
│  │ └──────────┘ │  │ └──────────┘ │  │          │  │        │ │
│  └──────────────┘  └──────────────┘  └──────────┘  └────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Blob Storage – Access Tiers:**

```
         BLOB ACCESS TIERS (Cost vs. Access Frequency)
         ─────────────────────────────────────────────
  Cost
  (Storage) │  HOT ──────────── COOL ──────────── ARCHIVE
  HIGH      │   │                  │                  │
            │   │ Frequent Access  │ Infrequent Access│ Rare Access
  LOW       │   │ Lowest latency   │ Higher latency   │ Hours rehydration
            └───┴──────────────────┴──────────────────┴─────────────►
                                                        Access Cost (HIGH→)
```

**Data Redundancy Options:**

```
  LRS   ─── 3 copies in 1 datacenter (same region)
  ZRS   ─── 3 copies across 3 Availability Zones (same region)
  GRS   ─── LRS + async replication to secondary region (6 copies total)
  GZRS  ─── ZRS + async replication to secondary region (6 copies total)
  RA-GRS ── GRS + read access to secondary region
```

**Blob Storage Types:**

| Type | Best For | Max Size |
|------|----------|----------|
| Block Blob | Files, images, videos, backups | ~190.7 TiB |
| Append Blob | Logging, audit trails | ~195 GiB |
| Page Blob | VHDs, random read/write workloads | 8 TiB |

---

### 1.2 End-to-End Tutorial

#### Step 1: Create a Storage Account

```bash
# Using Azure CLI
az storage account create \
  --name mystorageaccount \
  --resource-group myResourceGroup \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Hot
```

**Portal Path:** Azure Portal → Create a resource → Storage → Storage account

#### Step 2: Create a Blob Container

```bash
az storage container create \
  --name mycontainer \
  --account-name mystorageaccount \
  --public-access blob
```

**Access levels:**
- `off`  – Private (default, recommended)
- `blob` – Public read access for blobs only
- `container` – Public read access for container and blobs

#### Step 3: Upload a Blob (Python SDK)

```python
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os

# Connection string from Azure Portal → Storage Account → Access Keys
connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

# Create client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get container client
container_client = blob_service_client.get_container_client("mycontainer")

# Upload file
with open("./resume.pdf", "rb") as data:
    blob_client = container_client.upload_blob(
        name="resumes/resume.pdf",
        data=data,
        overwrite=True
    )

print(f"Blob URL: {blob_client.url}")
```

#### Step 4: Generate a SAS Token (Secure Access)

```python
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

sas_token = generate_blob_sas(
    account_name="mystorageaccount",
    container_name="mycontainer",
    blob_name="resumes/resume.pdf",
    account_key=os.environ["AZURE_STORAGE_KEY"],
    permission=BlobSasPermissions(read=True),
    expiry=datetime.utcnow() + timedelta(hours=1)
)

blob_url_with_sas = f"https://mystorageaccount.blob.core.windows.net/mycontainer/resumes/resume.pdf?{sas_token}"
```

#### Step 5: List, Download, and Delete Blobs

```python
# List blobs in container
for blob in container_client.list_blobs(name_starts_with="resumes/"):
    print(f"Blob: {blob.name}, Size: {blob.size} bytes")

# Download blob
blob_client = blob_service_client.get_blob_client(
    container="mycontainer",
    blob="resumes/resume.pdf"
)
with open("./downloaded_resume.pdf", "wb") as f:
    f.write(blob_client.download_blob().readall())

# Delete blob
blob_client.delete_blob()
```

#### Step 6: Lifecycle Management Policy

```json
{
  "rules": [
    {
      "name": "MoveToCoolAfter30Days",
      "enabled": true,
      "type": "Lifecycle",
      "definition": {
        "filters": { "blobTypes": ["blockBlob"], "prefixMatch": ["resumes/"] },
        "actions": {
          "baseBlob": {
            "tierToCool": { "daysAfterModificationGreaterThan": 30 },
            "tierToArchive": { "daysAfterModificationGreaterThan": 90 },
            "delete": { "daysAfterModificationGreaterThan": 365 }
          }
        }
      }
    }
  ]
}
```

#### Complete Architecture Flow – Blob Storage with this Application

```
  Client App / FastAPI (uphires_v1)
         │
         │ Upload resume file
         ▼
  Azure Blob Storage ──► Container: "resumes"
         │                    │
         │ Event triggers      │ Blob stored with metadata
         ▼                    ▼
  Azure Event Grid      Lifecycle Policy
         │               (Hot → Cool → Archive)
         │
         ▼
  Azure Function / Queue ──► Resume Parser API
                                    │
                                    ▼
                            MongoDB Atlas / CosmosDB
```

---

### 1.3 Interview Questions

**Q1. What are the four types of Azure Storage services, and when would you use each?**

> **Answer:** Azure Storage offers four services:
> - **Blob Storage** – Unstructured data (files, images, videos, backups). Use for storing resumes, media files, or large datasets.
> - **Azure Files** – Fully managed SMB/NFS file shares. Use when applications need a shared file system accessible from VMs or on-premises.
> - **Queue Storage** – Message queuing for async communication between components. Use to decouple services (e.g., trigger resume parsing after upload).
> - **Table Storage** – NoSQL key-value store for structured data without complex joins. Use for lightweight metadata storage.

---

**Q2. Explain the difference between LRS, ZRS, GRS, and RA-GRS redundancy options.**

> **Answer:**
> - **LRS (Locally Redundant Storage):** 3 synchronous copies within a single datacenter. Cheapest option but no protection against datacenter failure.
> - **ZRS (Zone-Redundant Storage):** 3 copies spread across 3 Availability Zones in the same region. Protects against zone failures.
> - **GRS (Geo-Redundant Storage):** LRS in primary region + async replication to secondary region (6 copies total). Protects against regional outages.
> - **RA-GRS (Read-Access GRS):** Same as GRS but grants read access to the secondary region endpoint even when primary is available.

---

**Q3. What is a Shared Access Signature (SAS) and when should you use it instead of an account key?**

> **Answer:** A SAS is a signed URI that grants restricted access to Azure Storage resources for a specified time period with specific permissions (read, write, delete, list). You should use SAS instead of sharing the storage account key because:
> - SAS tokens have expiry times, limiting exposure
> - They can be scoped to specific resources (container, blob)
> - Permissions can be restricted (e.g., read-only)
> - They can be revoked via a stored access policy
> Account keys give full control over the entire storage account and should never be shared externally.

---

**Q4. What are the three types of Blobs in Azure Blob Storage? Describe each.**

> **Answer:**
> - **Block Blob:** Optimized for uploading large amounts of data efficiently. Composed of blocks; each block can be uploaded independently and committed together. Ideal for files, images, videos.
> - **Append Blob:** Optimized for append operations. Supports adding data to the end without modifying existing content. Ideal for logging and audit trails.
> - **Page Blob:** Optimized for random read/write operations with 512-byte pages. Used as the underlying storage for Azure Virtual Machine disks (VHDs).

---

**Q5. How do you implement blob lifecycle management, and why is it important?**

> **Answer:** Blob lifecycle management uses JSON-based policies to automatically transition blobs between access tiers (Hot → Cool → Archive) or delete them based on rules like days since last modification or creation. It is important because:
> - Reduces storage costs by moving infrequently accessed data to cheaper tiers
> - Automates data governance and retention policies
> - Eliminates manual cleanup tasks
> Policies are configured at the storage account level and apply to block blobs and append blobs.

---

**Q6. What is the difference between Hot, Cool, and Archive access tiers?**

> **Answer:**
> - **Hot:** Optimized for frequently accessed data. Highest storage cost, lowest access cost. Best for active data.
> - **Cool:** Optimized for infrequently accessed data (at least 30 days). Lower storage cost, higher access cost. Best for backups and older data. Minimum storage duration of 30 days.
> - **Archive:** Lowest storage cost but data is offline. Retrieval requires "rehydration" which takes hours. Minimum storage duration of 180 days. Best for long-term archives and compliance data.

---

**Q7. How would you secure access to Azure Blob Storage?**

> **Answer:** Multiple layers of security:
> - **Azure RBAC** – Assign roles (Storage Blob Data Reader, Contributor) to identities (users, service principals, managed identities)
> - **SAS Tokens** – Scoped, time-limited access without sharing keys
> - **Private Endpoints** – Restrict access to a VNet, blocking public internet access
> - **Firewall Rules** – Allow only specific IP ranges or VNets
> - **Encryption** – Data encrypted at rest (AES-256) and in transit (HTTPS/TLS)
> - **Managed Identity** – Applications authenticate using Azure AD without storing credentials
> - **Azure Defender for Storage** – Threat detection for anomalous access patterns

---

**Q8. What is Azure Data Lake Storage Gen2 and how does it differ from standard Blob Storage?**

> **Answer:** Azure Data Lake Storage Gen2 (ADLS Gen2) is built on top of Azure Blob Storage with a hierarchical namespace (HNS) enabled. Key differences:
> - **Hierarchical Namespace:** True directory structure with atomic rename/delete operations on directories (not simulated via prefixes)
> - **Fine-grained ACLs:** POSIX-compliant ACLs on files and directories (not just RBAC at container level)
> - **Analytics Optimized:** Better performance for big data analytics workloads (Spark, Hadoop, Azure Synapse)
> - **Same pricing** as Blob Storage with HNS feature enabled on the storage account

---

**Q9. Explain Azure Storage static website hosting.**

> **Answer:** Azure Blob Storage can host static websites (HTML, CSS, JS, images) directly without a web server. When enabled:
> - A special `$web` container is created
> - Files uploaded to `$web` are served via a public endpoint (`https://<account>.z6.web.core.windows.net`)
> - You can configure a custom domain and use Azure CDN for HTTPS and global distribution
> - No server-side processing (no PHP/Python/Node); for dynamic content, use Azure Functions or App Service
> - Costs are just blob storage + bandwidth, making it extremely cheap for static sites

---

**Q10. How does Azure Storage handle consistency? What are the consistency guarantees?**

> **Answer:** Azure Blob Storage provides **strong consistency** for all operations:
> - After a successful write or delete operation, any subsequent read immediately reflects the latest state
> - This applies to all blob types in all regions
> - For geo-replicated accounts (GRS/RA-GRS), the secondary region provides **eventual consistency** – reads from the secondary may lag behind the primary
> - Azure Tables follows the same strong consistency model within a single partition
> This differs from some other cloud providers where eventual consistency is the default, making Azure Storage predictable for application development.

---

## 2. Azure App Service

### 2.1 Overview & Architecture

Azure App Service is a fully managed platform for building, deploying, and scaling web apps, REST APIs, and mobile backends without managing infrastructure.

**Service Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     AZURE APP SERVICE                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   App Service Plan                       │   │
│  │    (Defines: Region, VM Size, OS, Pricing Tier)          │   │
│  │                                                          │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │   │  Web App │  │ API App  │  │ Function │             │   │
│  │   │ (FastAPI)│  │ (REST)   │  │  App     │             │   │
│  │   └──────────┘  └──────────┘  └──────────┘             │   │
│  │                                                          │   │
│  │   ┌──────────────────────────────────────────────────┐  │   │
│  │   │          Deployment Slots                         │  │   │
│  │   │  [Production]  [Staging]  [Testing]               │  │   │
│  │   └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Auto-Scale   │  │  Custom      │  │  Managed Identity    │  │
│  │ Rules        │  │  Domain/SSL  │  │  (No creds stored)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Supported Runtimes:**

```
  ┌──────────┬────────────────────────────────────────────────┐
  │  Runtime │ Versions Supported                             │
  ├──────────┼────────────────────────────────────────────────┤
  │ Python   │ 3.8, 3.9, 3.10, 3.11, 3.12                    │
  │ Node.js  │ 18, 20 LTS                                     │
  │ .NET     │ 6, 7, 8                                        │
  │ Java     │ 8, 11, 17, 21                                  │
  │ PHP      │ 8.0, 8.1, 8.2                                  │
  │ Ruby     │ 3.2 (Linux only)                               │
  │ Custom   │ Docker containers                              │
  └──────────┴────────────────────────────────────────────────┘
```

**Pricing Tiers:**

```
  Free (F1)    ─── Shared, 60 min/day, no custom domain
  Shared (D1)  ─── Shared, 240 min/day, custom domain
  Basic (B1-3) ─── Dedicated VMs, manual scale (1-3 instances)
  Standard     ─── Auto-scale (10 instances), slots (5), VNet
  Premium      ─── Auto-scale (30), slots (20), VNet, private endpoints
  Isolated     ─── App Service Environment (ASE), dedicated VNet
```

---

### 2.2 End-to-End Tutorial

#### Step 1: Deploy the FastAPI Application to Azure App Service

**Option A – Using Azure CLI (ZIP deploy):**

```bash
# Login
az login

# Create Resource Group
az group create --name uphires-rg --location eastus

# Create App Service Plan (Linux, Python)
az appservice plan create \
  --name uphires-plan \
  --resource-group uphires-rg \
  --sku B1 \
  --is-linux

# Create Web App
az webapp create \
  --name uphires-api \
  --resource-group uphires-rg \
  --plan uphires-plan \
  --runtime "PYTHON:3.11"

# Set startup command for FastAPI/Uvicorn
az webapp config set \
  --name uphires-api \
  --resource-group uphires-rg \
  --startup-file "uvicorn main:app --host 0.0.0.0 --port 8000"

# Deploy from local directory (zip deploy)
cd /path/to/uphires_v1
zip -r app.zip . -x "*.git*" -x "__pycache__/*"
az webapp deploy \
  --name uphires-api \
  --resource-group uphires-rg \
  --src-path app.zip
```

**Option B – Using GitHub Actions (CI/CD):**

```yaml
# .github/workflows/azure-deploy.yml
name: Deploy to Azure App Service

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: python -m pytest

      - name: Deploy to Azure Web App
        uses: azure/webapps-deploy@v2
        with:
          app-name: 'uphires-api'
          publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
          package: .
```

#### Step 2: Configure Application Settings (Environment Variables)

```bash
az webapp config appsettings set \
  --name uphires-api \
  --resource-group uphires-rg \
  --settings \
    MONGODB_URI="mongodb+srv://..." \
    GROQ_API_KEY="gsk_..." \
    AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..." \
    PYTHON_ENV="production"
```

#### Step 3: Enable Managed Identity (Secure Azure Service Access)

```bash
# Enable system-assigned managed identity
az webapp identity assign \
  --name uphires-api \
  --resource-group uphires-rg

# Grant identity access to Blob Storage
PRINCIPAL_ID=$(az webapp identity show \
  --name uphires-api \
  --resource-group uphires-rg \
  --query principalId -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/<sub-id>/resourceGroups/uphires-rg/providers/Microsoft.Storage/storageAccounts/mystorageaccount"
```

**Using Managed Identity in Code (no credentials stored):**

```python
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# DefaultAzureCredential auto-detects the environment:
# - Managed Identity when running on Azure
# - Local developer credentials when running locally
credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url="https://mystorageaccount.blob.core.windows.net",
    credential=credential
)
```

#### Step 4: Configure Auto-Scaling

```bash
# Create autoscale settings
az monitor autoscale create \
  --resource-group uphires-rg \
  --resource uphires-plan \
  --resource-type Microsoft.Web/serverFarms \
  --name uphires-autoscale \
  --min-count 1 \
  --max-count 10 \
  --count 1

# Add scale-out rule (CPU > 70%)
az monitor autoscale rule create \
  --resource-group uphires-rg \
  --autoscale-name uphires-autoscale \
  --scale out 2 \
  --condition "Percentage CPU > 70 avg 5m"

# Add scale-in rule (CPU < 30%)
az monitor autoscale rule create \
  --resource-group uphires-rg \
  --autoscale-name uphires-autoscale \
  --scale in 1 \
  --condition "Percentage CPU < 30 avg 5m"
```

#### Step 5: Deployment Slots (Blue-Green Deployment)

```
  ┌──────────────────────────────────────────────────┐
  │              Deployment Slot Strategy            │
  │                                                  │
  │   STAGING SLOT          PRODUCTION SLOT          │
  │   (new version)         (live traffic)           │
  │        │                      │                  │
  │        │  ──── SWAP ────►     │                  │
  │        │   (zero downtime)    │                  │
  │        │                      │                  │
  │   uphires-api/staging    uphires-api (main)      │
  └──────────────────────────────────────────────────┘
```

```bash
# Create staging slot
az webapp deployment slot create \
  --name uphires-api \
  --resource-group uphires-rg \
  --slot staging

# Deploy to staging
az webapp deploy \
  --name uphires-api \
  --resource-group uphires-rg \
  --slot staging \
  --src-path app.zip

# Test staging: https://uphires-api-staging.azurewebsites.net

# Swap staging to production (zero downtime)
az webapp deployment slot swap \
  --name uphires-api \
  --resource-group uphires-rg \
  --slot staging \
  --target-slot production
```

#### Step 6: Configure Custom Domain and SSL

```bash
# Add custom domain
az webapp config hostname add \
  --webapp-name uphires-api \
  --resource-group uphires-rg \
  --hostname api.uphires.com

# Bind free managed certificate
az webapp config ssl bind \
  --name uphires-api \
  --resource-group uphires-rg \
  --certificate-thumbprint <thumbprint> \
  --ssl-type SNI
```

#### Complete Deployment Architecture

```
  Developer / CI/CD Pipeline
         │
         │ git push / GitHub Actions
         ▼
  Azure App Service (uphires-api)
  ┌─────────────────────────────┐
  │  Staging Slot               │
  │  (test & validate)          │
  │         │                   │
  │     SWAP│                   │
  │         ▼                   │
  │  Production Slot            │
  │  FastAPI + Uvicorn          │
  │  (Auto-scaled 1-10 VMs)     │
  └─────────────────────────────┘
         │         │
         ▼         ▼
  MongoDB       Azure Blob
  Atlas         Storage
                (resumes)
         │
         ▼
  Azure Monitor / App Insights
  (Logging, Metrics, Alerts)
```

---

### 2.3 Interview Questions

**Q1. What is Azure App Service and what types of applications can it host?**

> **Answer:** Azure App Service is a fully managed PaaS (Platform as a Service) for hosting web applications, REST APIs, and mobile backends. It abstracts away server management, OS patching, and infrastructure. It supports:
> - **Web Apps** – Any language runtime (Python, Node.js, .NET, Java, PHP, Ruby)
> - **API Apps** – RESTful APIs with built-in Swagger/OpenAPI support
> - **Mobile Apps** – Backend for iOS/Android apps
> - **Logic Apps** – Workflow automation (now separate service)
> - **Function Apps** – Serverless event-driven compute
> It runs on Linux or Windows and supports custom Docker containers.

---

**Q2. What is an App Service Plan? How does it affect cost and performance?**

> **Answer:** An App Service Plan defines the compute resources (VM size, OS, region) shared by one or more App Service apps. It determines:
> - **Cost:** You pay for the plan regardless of whether apps are running
> - **Scale:** The plan's tier limits max instances, deployment slots, and features
> - **Sharing:** Multiple apps on the same plan share the same VMs (useful for cost optimization but can cause resource contention)
> 
> Tiers: Free → Shared → Basic → Standard → Premium → Isolated (ASE)
> 
> Best practice: Don't mix production and dev/test apps on the same plan to avoid resource contention.

---

**Q3. Explain deployment slots and how they enable zero-downtime deployments.**

> **Answer:** Deployment slots are live environments with their own hostnames within the same App Service. The default (production) slot is always present; additional slots can be added based on the pricing tier.
> 
> **Zero-downtime flow:**
> 1. Deploy new version to staging slot
> 2. Test/validate staging (it has its own URL)
> 3. **Swap** staging and production – Azure swaps the underlying VMs, so traffic instantly switches with no restart required
> 4. If issues arise, swap back immediately
> 
> Slots also support **slot-sticky settings** (app settings that don't swap), allowing different database connections per environment.

---

**Q4. How does Azure App Service handle scaling?**

> **Answer:** App Service supports two types of scaling:
> - **Scale Up (Vertical):** Move to a larger pricing tier (more CPU/RAM per VM). Requires a brief restart.
> - **Scale Out (Horizontal):** Add more VM instances. Can be manual or automatic.
> 
> **Autoscale** (Standard tier and above) allows rules based on:
> - Metrics: CPU percentage, memory usage, HTTP queue length, custom metrics
> - Schedules: Scale up during business hours, down overnight
> - Predictive autoscale: ML-based pre-scaling based on historical patterns
> 
> Note: Free and Shared tiers do NOT support autoscaling.

---

**Q5. What is a Managed Identity, and why is it preferred over storing credentials in app settings?**

> **Answer:** A Managed Identity is an Azure AD identity automatically managed by Azure for an App Service (or other Azure resource). There are two types:
> - **System-Assigned:** Created and tied to the specific app; deleted when the app is deleted
> - **User-Assigned:** Created independently and can be assigned to multiple resources
> 
> **Why preferred over credentials:**
> - No secrets to rotate, store, or accidentally expose in code/logs
> - Azure handles token refresh automatically
> - Works with RBAC for fine-grained access control to other Azure services (Storage, Key Vault, Cognitive Services)
> - Eliminates the risk of credential leakage in source code

---

**Q6. How do you configure environment variables and connection strings in App Service?**

> **Answer:** App Service provides **Application Settings** (key-value pairs exposed as environment variables) and **Connection Strings** (injected differently per runtime):
> - Set via: Portal → Configuration → Application Settings / Connection Strings
> - Or via CLI: `az webapp config appsettings set`
> - App settings override values in `appsettings.json` or `.env` files
> - Settings can be **slot-specific** (sticky) or shared across slots
> - Sensitive values should be stored in **Azure Key Vault** with the app setting referencing the Key Vault secret (`@Microsoft.KeyVault(...)`)
> 
> For FastAPI: access via `os.environ["KEY_NAME"]` or `os.getenv("KEY_NAME")`

---

**Q7. What is App Service Environment (ASE) and when would you use it?**

> **Answer:** App Service Environment (ASE) is an isolated, single-tenant deployment of Azure App Service injected directly into a customer's Azure Virtual Network. Use cases:
> - **Regulatory compliance** – Complete network isolation with no shared infrastructure
> - **Very high scale** – Up to 100 instances (vs 30 for Premium)
> - **VNet integration** – Apps connect to on-premises resources via VPN/ExpressRoute with no outbound internet needed
> - **Custom IP addresses** – Dedicated inbound and outbound IPs
> 
> ASE costs significantly more (dedicated infrastructure) and should only be used when isolation is a business/compliance requirement.

---

**Q8. How does App Service handle application logging and monitoring?**

> **Answer:** App Service provides multiple logging options:
> - **Application Logging** – Python/app-level logs streamed to Blob Storage or filesystem
> - **Web Server Logging** – Raw HTTP request logs (IIS/nginx)
> - **Detailed Error Messages** – HTML error pages for 400+ HTTP errors
> - **Failed Request Tracing** – Detailed XML traces for failed requests
> 
> For production monitoring:
> - **Azure Monitor** – Metrics (CPU, memory, requests/sec, response time)
> - **Application Insights** – Distributed tracing, live metrics, dependency tracking, exception tracking
> - **Log Analytics** – Query logs with KQL (Kusto Query Language)
> - **Alerts** – Email/webhook notifications when thresholds are breached

---

**Q9. Explain VNet Integration in App Service. What problem does it solve?**

> **Answer:** VNet Integration allows outbound traffic from an App Service to flow through an Azure Virtual Network, enabling the app to reach resources inside a VNet (VMs, internal databases, on-premises via VPN/ExpressRoute). 
> 
> **Problem it solves:** By default, App Service runs on multi-tenant shared infrastructure and cannot reach resources inside a VNet. VNet Integration bridges this gap.
> 
> **Types:**
> - **Regional VNet Integration** (Standard+): App connects to a VNet in the same region via a dedicated subnet. Best for most use cases.
> - **Gateway-Required VNet Integration**: Connect to VNets in different regions or classic VNets (legacy).
> 
> Note: VNet Integration is for **outbound** traffic only. For inbound private access (blocking public access), use **Private Endpoints**.

---

**Q10. How would you implement CI/CD for an App Service deployed FastAPI application?**

> **Answer:** Full CI/CD pipeline for App Service:
> 1. **Source Control:** GitHub / Azure Repos as the trigger
> 2. **Build:** GitHub Actions or Azure DevOps Pipelines
>    - Run `pip install -r requirements.txt`
>    - Run unit tests (`pytest`)
>    - Run linting (`flake8`, `ruff`)
> 3. **Deploy to Staging Slot:** Using `azure/webapps-deploy` action with `slot: staging`
> 4. **Integration Tests:** Run smoke tests against the staging URL
> 5. **Swap to Production:** Use `az webapp deployment slot swap` in the pipeline
> 6. **Rollback:** If post-deployment tests fail, auto-swap back to previous version
> 
> Key GitHub Actions secrets: `AZURE_WEBAPP_PUBLISH_PROFILE` or `AZURE_CREDENTIALS` (service principal).

---

## 3. Azure AI Foundry

### 3.1 Overview & Architecture

Azure AI Foundry (formerly Azure AI Studio) is Microsoft's unified platform for building, evaluating, and deploying generative AI applications and models at enterprise scale.

**Platform Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      AZURE AI FOUNDRY                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    AI Foundry Hub                        │   │
│  │  (Central governance: security, connections, compute)    │   │
│  │                                                          │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │   │  Project A   │  │  Project B   │  │  Project C   │ │   │
│  │   │(Chat App)    │  │(RAG Pipeline)│  │(Fine-tuning) │ │   │
│  │   └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  Model       │  │  Prompt      │  │   Evaluation       │   │
│  │  Catalog     │  │  Flow        │  │   Framework        │   │
│  │              │  │  (Pipelines) │  │                    │   │
│  │ • OpenAI GPT │  │ • LLM chains │  │ • Groundedness     │   │
│  │ • Llama      │  │ • RAG flows  │  │ • Relevance        │   │
│  │ • Mistral    │  │ • Agents     │  │ • Coherence        │   │
│  │ • Phi series │  │              │  │ • Fluency          │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**

```
  AI FOUNDRY COMPONENTS
  ─────────────────────
  
  Model Catalog        ──► Browse, test, deploy 1600+ models
         │
  Prompt Flow          ──► Visual pipeline builder for LLM apps
         │
  Evaluation           ──► Assess model/app quality with AI metrics
         │
  Azure OpenAI         ──► GPT-4, GPT-4o, DALL-E, Whisper, Embeddings
         │
  Fine-tuning          ──► Customize models on your data
         │
  Connections          ──► Securely link to Storage, Search, APIs
         │
  Content Safety       ──► Filter harmful content in/out of LLM
         │
  Monitoring           ──► Track latency, cost, quality over time
```

---

### 3.2 End-to-End Tutorial

#### Step 1: Create an AI Foundry Hub and Project

```bash
# Install Azure AI CLI
pip install azure-ai-ml azure-ai-projects azure-identity

# Create Resource Group
az group create --name ai-foundry-rg --location eastus

# Create AI Hub (via Portal or bicep/ARM)
# Portal path: Azure AI Foundry portal (ai.azure.com)
#   → New Hub → Name: uphires-hub
#   → Connected resources: Storage, Key Vault, AI Services
```

#### Step 2: Deploy a Model from the Model Catalog

**Via Azure AI Foundry Portal:**

```
ai.azure.com
  → Select Hub → Select Project
  → Deployments → Model Catalog
  → Search: "gpt-4o"
  → Deploy → Deployment name: gpt4o-deployment
  → Version: 2024-08-06
  → Tokens per minute: 10,000
```

**Via Python SDK:**

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group_name="ai-foundry-rg",
    workspace_name="uphires-project"
)

# List available models
for model in ml_client.models.list():
    print(f"{model.name}: {model.version}")
```

#### Step 3: Build a RAG Application with Prompt Flow

**RAG (Retrieval-Augmented Generation) architecture used in uphires_v1:**

```
  User Query: "Find Python developers with 5 years experience"
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    PROMPT FLOW                          │
  │                                                         │
  │  1. Embed Query ──► Azure OpenAI Embeddings             │
  │         │                                               │
  │         ▼                                               │
  │  2. Vector Search ──► Azure AI Search / MongoDB Atlas   │
  │         │                                               │
  │         ▼                                               │
  │  3. Context Assembly ──► Top-K resume chunks            │
  │         │                                               │
  │         ▼                                               │
  │  4. LLM Generation ──► GPT-4o with context              │
  │         │                                               │
  │         ▼                                               │
  │  5. Response ──► Ranked candidate list + reasoning      │
  └─────────────────────────────────────────────────────────┘
```

**Python implementation using Azure AI Projects SDK:**

```python
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

# Initialize AI Project client
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str="<ai-foundry-project-connection-string>"
)

# Get the Azure OpenAI connection from the project
openai_connection = project_client.connections.get_default(
    connection_type=ConnectionType.AZURE_OPEN_AI
)

# Create OpenAI client using the project connection
openai_client = AzureOpenAI(
    api_key=openai_connection.key,
    azure_endpoint=openai_connection.endpoint_url,
    api_version="2024-08-01-preview"
)

# Chat completion
def search_candidates_with_llm(query: str, resume_context: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt4o-deployment",
        messages=[
            {
                "role": "system",
                "content": "You are a recruitment assistant. Analyze resumes and rank candidates."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nResume Data:\n{resume_context}\n\nRank the most relevant candidates."
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content
```

#### Step 4: Evaluate the AI Application

```python
from azure.ai.evaluation import evaluate, RelevanceEvaluator, GroundednessEvaluator

# Define evaluators
relevance_evaluator = RelevanceEvaluator(
    model_config={
        "azure_endpoint": "<endpoint>",
        "azure_deployment": "gpt4o-deployment",
        "api_key": "<key>",
        "api_version": "2024-08-01-preview"
    }
)

groundedness_evaluator = GroundednessEvaluator(
    model_config={...}
)

# Run evaluation on a test dataset
results = evaluate(
    data="./test_queries.jsonl",  # {"query": "...", "response": "...", "context": "..."}
    evaluators={
        "relevance": relevance_evaluator,
        "groundedness": groundedness_evaluator
    },
    output_path="./evaluation_results.json"
)

print(f"Average Relevance: {results['metrics']['relevance.score']}")
print(f"Average Groundedness: {results['metrics']['groundedness.score']}")
```

#### Step 5: Deploy as an Online Endpoint

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="uphires-rag-endpoint",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="uphires-rag-endpoint",
    model="azureml:uphires-rag-model:1",
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
```

#### Complete AI Foundry Flow:

```
  ┌────────────────────────────────────────────────────────┐
  │                  DEVELOPMENT FLOW                      │
  │                                                        │
  │  1. EXPLORE      ──► Model Catalog (test models)       │
  │  2. BUILD        ──► Prompt Flow (visual pipeline)     │
  │  3. EVALUATE     ──► AI Evaluations (quality metrics)  │
  │  4. FINE-TUNE    ──► Custom model training (optional)  │
  │  5. DEPLOY       ──► Managed Online Endpoint           │
  │  6. MONITOR      ──► Application Insights + dashboards │
  └────────────────────────────────────────────────────────┘
```

---

### 3.3 Interview Questions

**Q1. What is Azure AI Foundry and how does it differ from Azure OpenAI Service?**

> **Answer:**
> - **Azure OpenAI Service** is a specific service providing access to OpenAI's models (GPT-4, DALL-E, Whisper, Embeddings) via REST APIs with Azure enterprise features (security, compliance, SLAs).
> - **Azure AI Foundry** (formerly Azure AI Studio) is the unified **development platform** for building, evaluating, deploying, and managing AI applications. It acts as the orchestration layer on top of Azure OpenAI and other models.
> 
> Foundry provides: Model Catalog (1600+ models including non-OpenAI), Prompt Flow (visual pipeline builder), built-in evaluation framework, fine-tuning capabilities, governance via Hubs and Projects, and integrated safety features. Think of Azure OpenAI as a component, and AI Foundry as the complete development environment.

---

**Q2. What is Prompt Flow in Azure AI Foundry and what are its main node types?**

> **Answer:** Prompt Flow is a visual development tool for building LLM-based application pipelines (workflows). It connects multiple steps into a Directed Acyclic Graph (DAG) and supports:
> 
> **Node Types:**
> - **LLM Node:** Call an LLM with a prompt template (supports Jinja2 templating)
> - **Python Node:** Execute arbitrary Python code for data transformation, API calls, custom logic
> - **Prompt Node:** Render and format prompt templates with variable injection
> - **Tool Node:** Pre-built integrations (Embedding, Vector Index Lookup, Content Safety)
> 
> Flows can be exported as Python code, tested locally, deployed as managed endpoints, and evaluated automatically. This is used for building RAG pipelines, chat applications, and agentic workflows.

---

**Q3. How does the Azure AI Foundry Model Catalog work? How do you select the right model?**

> **Answer:** The Model Catalog contains 1600+ models organized by:
> - **Provider:** OpenAI (GPT-4, GPT-4o), Meta (Llama 3), Mistral, Cohere, Microsoft (Phi-3), HuggingFace community models
> - **Task:** Chat completion, text generation, embeddings, image generation, speech-to-text, classification
> - **License:** Open (Apache 2.0, MIT) vs commercial
> 
> **Model selection criteria:**
> - **Performance vs cost:** GPT-4o > GPT-4 mini; use smaller models for simple tasks
> - **Latency requirements:** Phi-3-mini for edge/low-latency scenarios
> - **Data privacy:** Decide whether to use serverless API (shared) or deploy to dedicated compute
> - **Benchmarks:** Compare on MMLU, HumanEval, GSM8K based on your task type
> - **Fine-tuning support:** Check if the model supports fine-tuning if customization is needed

---

**Q4. What is the difference between a Hub and a Project in Azure AI Foundry?**

> **Answer:**
> - **Hub:** The central governance unit managed by IT/admin teams. It defines:
>   - Shared compute, storage, and network configurations
>   - Security settings (private endpoints, managed network)
>   - Connected resources (Azure AI Services, Storage, Key Vault, Search)
>   - Cost management and access policies
>   
> - **Project:** A workspace within a Hub where data scientists and developers work. It:
>   - Inherits Hub's security and connections
>   - Contains experiments, flows, evaluations, deployments
>   - Has its own access controls (team-level isolation)
>   - Can be created/deleted without affecting other projects or the Hub
>   
> Think of Hub as the "department" and Project as the "team workspace" within that department.

---

**Q5. How does Azure AI Foundry support model fine-tuning? What are the prerequisites?**

> **Answer:** Fine-tuning customizes a base model with domain-specific data using supervised learning:
> 
> **Supported models for fine-tuning:** GPT-4o mini, GPT-3.5 Turbo, Llama 2/3, Phi-3
> 
> **Prerequisites:**
> - Training data in JSONL format: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
> - Minimum ~50-100 high-quality examples (more is better, typically 500-1000+)
> - Validation dataset (10-20% of training size)
> 
> **Process:**
> 1. Upload training/validation data to AI Foundry
> 2. Select base model → Fine-tuning → Configure hyperparameters (epochs, learning rate, batch size)
> 3. Submit fine-tuning job (compute managed by Azure)
> 4. Evaluate fine-tuned model on held-out test set
> 5. Deploy fine-tuned model as a deployment
> 
> Use fine-tuning when prompt engineering alone cannot achieve required accuracy for domain-specific tasks.

---

**Q6. What evaluation metrics does Azure AI Foundry provide for generative AI applications?**

> **Answer:** Azure AI Foundry provides AI-assisted evaluation metrics using an LLM-as-judge approach:
> 
> **Quality Metrics (require reference/context):**
> - **Groundedness** (1-5): Does the response factually align with the provided context?
> - **Relevance** (1-5): How well does the response address the query?
> - **Coherence** (1-5): Is the response logically structured and easy to read?
> - **Fluency** (1-5): Is the response grammatically correct and natural?
> - **Similarity** (1-5): How similar is the response to the ground truth answer?
> 
> **Safety Metrics:**
> - **Violence, Sexual, Self-harm, Hate** content scores
> - **Jailbreak** detection (prompt injection attempts)
> - **Protected material** detection
> 
> **Custom Evaluators:** Define your own criteria using Python or LLM-based evaluation logic.

---

**Q7. How does Azure AI Foundry handle content safety and responsible AI?**

> **Answer:** Azure AI Foundry integrates **Azure AI Content Safety** at multiple levels:
> 
> - **Input filtering:** Detect and block harmful prompts before they reach the LLM (violence, sexual content, hate speech, self-harm)
> - **Output filtering:** Filter LLM responses that contain harmful content
> - **Jailbreak detection:** Identify prompt injection attempts trying to bypass safety guidelines
> - **Groundedness detection:** Flag hallucinated or ungrounded responses in RAG scenarios
> - **Custom blocklists:** Add domain-specific prohibited terms
> 
> **Responsible AI Dashboard:** Provides fairness analysis, error analysis, and model interpretability for ML models.
> 
> Content Safety is applied via **Safety System Messages** added to deployments and **Content Filters** configured at the deployment level.

---

**Q8. What are AI Agents in Azure AI Foundry and how do they differ from basic chat completions?**

> **Answer:**
> - **Basic chat completion:** Single LLM call, stateless, no tools, returns text response directly.
> 
> - **AI Agent (Azure AI Agent Service):** An orchestrated system where the LLM can autonomously:
>   - Use **tools** (code interpreter, Bing search, Azure AI Search, custom functions)
>   - Maintain **conversation threads** (stateful, multi-turn)
>   - **Plan and execute** multi-step tasks iteratively
>   - Call external APIs via **function calling**
>   - Execute code in a sandboxed environment
> 
> **Example for uphires_v1:** An AI Agent could receive "Find top Python developers in Bangalore with MongoDB experience," autonomously query the database tool, filter results, rank them, and return structured output — without pre-scripted logic for each step.

---

**Q9. How do you monitor and observe a generative AI application deployed via AI Foundry?**

> **Answer:** AI Foundry integrates with Azure Monitor and Application Insights:
> 
> **Built-in metrics tracked:**
> - Token usage (prompt tokens, completion tokens, total tokens per request)
> - Request latency (p50, p95, p99)
> - Error rates (4xx, 5xx, throttling)
> - Model deployment availability
> 
> **Online evaluation:** Continuously evaluate live traffic using sampled requests
> 
> **Tracing:** Distributed traces for multi-step Prompt Flows (see each node's inputs/outputs)
> 
> **Log Analytics integration:** Query logs with KQL to analyze patterns
> 
> **Alerts:** Configure alerts when cost exceeds budget, latency spikes, or quality scores drop below thresholds
> 
> **Cost management:** Track token consumption per deployment, set quotas

---

**Q10. What is the Azure AI Foundry connection system and why is it important?**

> **Answer:** Connections in AI Foundry are securely stored credentials/endpoint references to external services, centrally managed at the Hub level and shared with Projects:
> 
> **Supported connection types:**
> - Azure OpenAI (endpoint + key or managed identity)
> - Azure AI Search (for vector search in RAG)
> - Azure Blob Storage (for data ingestion)
> - Custom connections (any API with URL + headers)
> - Serverless API connections (model catalog deployments)
> 
> **Why important:**
> - Credentials are stored in the Hub's Key Vault, never exposed in code
> - Projects inherit Hub connections without needing to store secrets
> - Connection names are referenced in Prompt Flows, making them portable
> - Enables Managed Identity authentication — no keys in application code
> - Centralized rotation: update the connection once, all projects benefit

---

## 4. Azure AI Services

### 4.1 Overview & Architecture

Azure AI Services (formerly Azure Cognitive Services) is a collection of pre-built AI capabilities accessible via REST APIs and SDKs, allowing developers to add AI to applications without ML expertise.

**Service Categories:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      AZURE AI SERVICES                          │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   LANGUAGE      │  │    VISION       │  │    SPEECH       │ │
│  │                 │  │                 │  │                 │ │
│  │ • Language      │  │ • Computer      │  │ • Speech-to-    │ │
│  │   Understanding │  │   Vision        │  │   Text          │ │
│  │ • Text Analytics│  │ • Face API      │  │ • Text-to-      │ │
│  │ • Translator    │  │ • Form          │  │   Speech        │ │
│  │ • QnA Maker     │  │   Recognizer    │  │ • Speaker       │ │
│  │ • Azure OpenAI  │  │ • Custom Vision │  │   Recognition   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   DECISION      │  │    SEARCH       │  │    DOCUMENT     │ │
│  │                 │  │                 │  │ INTELLIGENCE    │ │
│  │ • Anomaly       │  │ • Bing Web      │  │                 │ │
│  │   Detector      │  │   Search        │  │ • Document      │ │
│  │ • Content       │  │ • Bing News     │  │   Analysis      │ │
│  │   Moderator     │  │ • Bing Image    │  │ • Pre-built     │ │
│  │ • Personalizer  │  │   Search        │  │   Models        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Request Flow Architecture:**

```
  Application Code
       │
       │ HTTPS REST call
       │ Headers: Ocp-Apim-Subscription-Key / Bearer token
       ▼
  Azure AI Services Endpoint
  (region-specific: eastus.api.cognitive.microsoft.com)
       │
       │ Routes to specific service
       ▼
  ┌─────────────────────────────────────────────────┐
  │          AI Processing Engine                   │
  │  (Language | Vision | Speech | Decision)        │
  └─────────────────────────────────────────────────┘
       │
       │ JSON response
       ▼
  Application Code processes result
```

---

### 4.2 End-to-End Tutorial

#### Step 1: Create an Azure AI Services Resource

```bash
# Create a multi-service AI Services resource
az cognitiveservices account create \
  --name uphires-ai-services \
  --resource-group uphires-rg \
  --kind CognitiveServices \
  --sku S0 \
  --location eastus \
  --yes

# Get the endpoint and key
az cognitiveservices account show \
  --name uphires-ai-services \
  --resource-group uphires-rg \
  --query "properties.endpoint"

az cognitiveservices account keys list \
  --name uphires-ai-services \
  --resource-group uphires-rg
```

#### Step 2: Azure AI Language Service – Text Analytics for Resume Processing

```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os

# Initialize client
endpoint = os.environ["AZURE_AI_ENDPOINT"]
key = os.environ["AZURE_AI_KEY"]

client = TextAnalyticsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Extract key phrases from a resume
def extract_skills_from_resume(resume_text: str) -> dict:
    documents = [resume_text]

    # Key phrase extraction
    key_phrases_result = client.extract_key_phrases(documents)
    key_phrases = key_phrases_result[0].key_phrases if not key_phrases_result[0].is_error else []

    # Named entity recognition (finds skills, companies, locations)
    entities_result = client.recognize_entities(documents)
    entities = []
    if not entities_result[0].is_error:
        entities = [
            {"text": entity.text, "category": entity.category, "confidence": entity.confidence_score}
            for entity in entities_result[0].entities
        ]

    # Sentiment analysis
    sentiment_result = client.analyze_sentiment(documents)
    sentiment = sentiment_result[0].sentiment if not sentiment_result[0].is_error else "unknown"

    return {
        "key_phrases": key_phrases,
        "entities": entities,
        "sentiment": sentiment
    }

# Example usage
resume = """
Experienced Python developer with 5 years of expertise in FastAPI, MongoDB, 
and machine learning. Worked at Microsoft and Google on cloud infrastructure projects.
Currently based in Bangalore, India.
"""
result = extract_skills_from_resume(resume)
print(result)
```

#### Step 3: Azure Document Intelligence – Parse Resume PDFs

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import os

di_client = DocumentIntelligenceClient(
    endpoint=os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"])
)

def parse_resume_pdf(pdf_path: str) -> dict:
    with open(pdf_path, "rb") as f:
        poller = di_client.begin_analyze_document(
            model_id="prebuilt-document",  # or "prebuilt-resume" if available
            analyze_request=f,
            content_type="application/pdf"
        )
    
    result = poller.result()
    
    extracted_data = {}
    
    # Extract key-value pairs
    for kv_pair in result.key_value_pairs or []:
        if kv_pair.key and kv_pair.value:
            extracted_data[kv_pair.key.content] = kv_pair.value.content
    
    # Extract all text by page
    full_text = ""
    for page in result.pages or []:
        for line in page.lines or []:
            full_text += line.content + "\n"
    
    return {"fields": extracted_data, "full_text": full_text}

# Use this to enhance the existing resume parser in uphires_v1
parsed = parse_resume_pdf("./resume.pdf")
print(parsed["fields"])
```

#### Step 4: Azure AI Translator – Multi-language Resume Processing

```python
import requests
import uuid
import os

def translate_resume(text: str, target_language: str = "en") -> str:
    """Translate resume text to English for processing"""
    endpoint = "https://api.cognitive.microsofttranslator.com"
    key = os.environ["AZURE_TRANSLATOR_KEY"]
    location = "eastus"

    url = f"{endpoint}/translate"
    params = {"api-version": "3.0", "to": target_language}
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }
    body = [{"text": text}]

    response = requests.post(url, params=params, headers=headers, json=body)
    response.raise_for_status()
    return response.json()[0]["translations"][0]["text"]
```

#### Step 5: Azure AI Speech – Audio Resume / Transcription

```python
import azure.cognitiveservices.speech as speechsdk
import os

def transcribe_audio_resume(audio_file_path: str) -> str:
    """Transcribe audio resume to text"""
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ["AZURE_SPEECH_KEY"],
        region=os.environ["AZURE_SPEECH_REGION"]
    )
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.AudioConfig(filename=audio_file_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        raise Exception(f"Speech recognition failed: {result.reason}")
```

#### Step 6: Azure Computer Vision – Extract Text from Resume Images

```python
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import os

vision_client = ImageAnalysisClient(
    endpoint=os.environ["AZURE_VISION_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_VISION_KEY"])
)

def extract_text_from_resume_image(image_path: str) -> str:
    """Extract text from resume saved as image (JPEG/PNG)"""
    with open(image_path, "rb") as f:
        result = vision_client.analyze(
            image_data=f.read(),
            visual_features=[VisualFeatures.READ]  # OCR
        )

    extracted_text = ""
    if result.read and result.read.blocks:
        for block in result.read.blocks:
            for line in block.lines:
                extracted_text += line.text + "\n"

    return extracted_text
```

#### Complete AI Services Integration Architecture for uphires_v1:

```
  Resume Input (PDF / DOCX / Image / Audio)
         │
         ├──► PDF/DOCX ──► Azure Document Intelligence ──► Structured Text
         │
         ├──► Image ───► Azure Computer Vision (OCR) ──► Text
         │
         ├──► Audio ───► Azure Speech-to-Text ──────────► Text
         │
         ▼
  Text Normalization
         │
         ├──► Non-English ──► Azure Translator ──► English Text
         │
         ▼
  Azure AI Language Service
  ├── Key Phrase Extraction ──► Skills, Technologies
  ├── Named Entity Recognition ──► Companies, Locations, Dates
  └── Text Classification ──► Job Level, Domain
         │
         ▼
  Sentence Transformers / Azure OpenAI Embeddings
         │
         ▼
  MongoDB Atlas (Vector Store) + Search Index
         │
         ▼
  uphires_v1 Search APIs (manual_search, vector_search, rag_search)
```

---

### 4.3 Interview Questions

**Q1. What is Azure AI Services (formerly Cognitive Services)? How is it structured?**

> **Answer:** Azure AI Services is a collection of pre-built, REST API-accessible AI capabilities that allow developers to integrate AI into applications without building or training ML models from scratch. It is structured into categories:
> - **Language:** Text analytics, translation, language understanding, QnA, Azure OpenAI
> - **Vision:** Image analysis, OCR, face detection, custom image classification, Document Intelligence
> - **Speech:** Speech-to-text, text-to-speech, speaker recognition, speech translation
> - **Decision:** Content moderation, anomaly detection, personalizer
> - **Search:** Bing Search APIs
> 
> Resources can be created as single-service (e.g., just Language) or multi-service (one key for all services). All services use standard HTTPS REST APIs with regional endpoints.

---

**Q2. What is Azure Document Intelligence (formerly Form Recognizer) and what are its model types?**

> **Answer:** Azure Document Intelligence is an AI service for extracting structured data from documents using ML-based OCR and form understanding. 
> 
> **Model Types:**
> - **Prebuilt models:** Ready-to-use for common document types: invoices, receipts, business cards, ID documents, tax forms (W-2), health insurance cards
> - **Layout model:** Extracts text, tables, selection marks, and document structure without domain-specific training
> - **General document model:** Key-value pairs, entities, and full text from any document
> - **Custom models:** Train on your own documents (labeling tool provided):
>   - Custom template model: Fixed-format documents
>   - Custom neural model: Variable-format documents
>   - Composed model: Route documents to appropriate custom models
> 
> For resume parsing in uphires_v1, the Layout or General Document model extracts text and structure from PDF resumes.

---

**Q3. Explain the difference between Azure AI Language Service capabilities: NER, Key Phrase Extraction, and Text Classification.**

> **Answer:**
> - **Named Entity Recognition (NER):** Identifies and categorizes entities in text into predefined types: Person, Organization, Location, DateTime, Quantity, URL, Email, Phone number. Example: "John worked at Microsoft in Seattle" → {John: Person, Microsoft: Organization, Seattle: Location}
> 
> - **Key Phrase Extraction:** Identifies the most important words and phrases that capture the main topics of the text. No predefined categories — extracts what the text is "about." Example: "Experienced Python developer in FastAPI and MongoDB" → ["Python developer", "FastAPI", "MongoDB"]
> 
> - **Text Classification:** Assigns the entire document (or sentence) to one or more user-defined categories. Requires training data with labeled examples (custom classification) or uses prebuilt categories. Example: Classify resume as "Software Engineering" vs "Marketing" vs "Finance"

---

**Q4. What is the difference between Azure AI Speech SDK and REST API? When would you use each?**

> **Answer:**
> - **REST API:**
>   - Simpler, stateless, good for short audio files (up to 60 seconds for recognition)
>   - Standard HTTP calls, language-agnostic
>   - Use for: Simple transcription of short clips, text-to-speech generation, batch transcription via the Batch Transcription REST API
>   - Best for server-side, non-real-time workloads
> 
> - **Speech SDK:**
>   - Stateful, handles streaming audio for real-time recognition
>   - Supports continuous recognition (long-form audio, live microphone)
>   - Handles connection management, reconnection, error recovery automatically
>   - Available for: C#, Python, JavaScript, Java, C++, Swift, Objective-C
>   - Best for: Real-time applications, voice interfaces, live transcription
> 
> For audio resume transcription in uphires_v1, the SDK with continuous recognition handles long audio files better.

---

**Q5. How does Azure AI Translator handle language detection vs. explicit source language specification?**

> **Answer:** Azure Translator supports both modes:
> 
> - **Auto-detection (recommended):** Omit the `from` parameter in the API call. The service uses ML to detect the source language from the text content. Returns detected language and confidence score in the response. Works for 130+ languages.
> 
> - **Explicit specification:** Provide `from=fr` (French) for performance (slightly faster, no detection step) and reliability when you know the source language.
> 
> **Important nuances:**
> - Auto-detection works on the first 1000 characters for efficiency
> - Detection accuracy decreases for very short text (< 5 words)
> - Use `detectLanguage` API separately if you only want to detect without translating
> - Transliteration (script conversion, e.g., Japanese hiragana to romaji) is a separate endpoint
> 
> For multilingual resume processing, auto-detection is recommended since candidate documents may come in any language.

---

**Q6. What is Custom Vision in Azure AI Services and what are its two types of projects?**

> **Answer:** Azure Custom Vision is a service for training custom image classification and object detection models without deep ML expertise, using a visual labeling interface.
> 
> **Two project types:**
> - **Image Classification:**
>   - **Multiclass:** Each image belongs to exactly one category. Example: Resume photo quality (good/blurry/missing)
>   - **Multilabel:** Each image can have multiple labels. Example: Resume document type (has-photo, has-signature, has-table)
>   
> - **Object Detection:** Locates and identifies multiple objects within a single image with bounding boxes. Example: Detecting sections in a resume image (header, experience, education, skills)
> 
> **Training requirements:** Minimum 15 images per class for classification. The model iterates and you can add more labeled images to improve accuracy. Models can be exported for offline use (TensorFlow Lite, CoreML, ONNX, Docker).

---

**Q7. What is the Azure AI Content Safety service and what are its key detection capabilities?**

> **Answer:** Azure AI Content Safety analyzes text and images to detect potentially harmful, offensive, or inappropriate content. Key capabilities:
> 
> **Text Analysis:**
> - Detects: Hate speech, sexual content, violence, self-harm
> - Returns severity scores (0-7 scale) per category
> - Supports custom blocklists for domain-specific terms
> - Jailbreak detection: Identifies attempts to bypass LLM safety guidelines
> - Protected material detection: Flags copyrighted text or code
> - Groundedness detection: Checks if LLM responses are grounded in provided context
> 
> **Image Analysis:**
> - Detects violent and sexual imagery with severity scores
> 
> **Integration:** Used as a middleware layer in front of LLM applications deployed via Azure AI Foundry or directly in content pipelines. For uphires_v1, it could filter inappropriate resume content before processing.

---

**Q8. How does Azure AI Services authentication work? What are the security best practices?**

> **Answer:** Azure AI Services supports multiple authentication methods:
> 
> **Authentication options:**
> - **API Key:** Simple key-based auth (`Ocp-Apim-Subscription-Key` header). Easy to use but requires secure storage.
> - **Azure AD / Managed Identity:** Use OAuth 2.0 bearer tokens. Recommended for production applications running on Azure.
> - **Service Principal:** For non-Azure environments needing Azure AD auth.
> 
> **Security best practices:**
> - Store API keys in **Azure Key Vault**, not in code or environment files checked into source control
> - Use **Managed Identity** in production to eliminate key management entirely
> - Assign **minimum RBAC permissions** (e.g., `Cognitive Services User` role)
> - Enable **Private Endpoints** to restrict service access to within your VNet
> - **Rotate keys** regularly; AI Services provides Key1 and Key2 for zero-downtime rotation
> - Enable **Azure Defender for AI** to detect abnormal API usage patterns

---

**Q9. What is the difference between Azure AI Services and Azure Machine Learning? When would you choose each?**

> **Answer:**
> - **Azure AI Services:** Pre-built, API-accessible AI capabilities (no training required). Choose when:
>   - You need a common AI task (OCR, translation, sentiment analysis, speech recognition)
>   - Time-to-value is critical (integrate in hours, not weeks)
>   - You don't have specialized training data or ML expertise
>   - The task fits standard categories (language, vision, speech, decision)
> 
> - **Azure Machine Learning:** End-to-end ML platform for custom model development. Choose when:
>   - You have unique data and need a custom model
>   - Pre-built models don't achieve sufficient accuracy for your use case
>   - You need full control over model architecture, training, and hyperparameters
>   - You're building traditional ML pipelines (regression, classification, clustering) on tabular data
>   - You need MLOps: versioning, pipelines, model registry, drift monitoring
> 
> **Often used together:** Use Azure AI Services for standard tasks and Azure ML for proprietary model training that feeds into AI-powered applications.

---

**Q10. How do you implement Azure AI Services at scale (high throughput, low latency)?**

> **Answer:** Key strategies for scaling Azure AI Services:
> 
> **Throughput optimization:**
> - **Batch processing:** Many services support batch APIs (process multiple documents in one call) — up to 5 documents for Language, 1000 for Translator
> - **Service tiers:** Move from Free (F0) to Standard (S0) tiers for higher rate limits; upgrade to Commitment pricing for predictable throughput
> - **Regional distribution:** Deploy in multiple regions and load balance across endpoints
> - **Request queuing:** Use Azure Service Bus or Queue Storage to buffer spikes
> 
> **Latency optimization:**
> - Use the **nearest region** to minimize network latency
> - Enable **Private Endpoints** for VNet-private access (removes public internet hops)
> - **Cache results:** Cache translation or entity extraction results for repeated/similar text
> - Use **asynchronous operations** for long-running tasks (Document Intelligence, batch transcription)
> 
> **Resilience:**
> - Implement **exponential backoff** with jitter for 429 (rate limit) responses
> - Distribute across **multiple Cognitive Services accounts** in different regions
> - Monitor with **Azure Monitor alerts** for latency spikes and error rate increases

---

## Summary

| Service | Key Skill | Common Interview Focus |
|---------|-----------|----------------------|
| **Azure Storage** | Blob, File, Queue, Table; redundancy; SAS; lifecycle | Security, data consistency, cost optimization |
| **Azure App Service** | PaaS deployment; slots; autoscale; managed identity | CI/CD, zero-downtime deploy, VNet integration |
| **Azure AI Foundry** | Model Catalog; Prompt Flow; evaluation; fine-tuning | RAG pipelines, evaluation metrics, safety |
| **Azure AI Services** | Language, Vision, Speech, Document Intelligence | Authentication, scaling, service selection |

---

## Quick Reference – Key Azure CLI Commands

```bash
# Storage
az storage account create --name <name> --resource-group <rg> --sku Standard_LRS
az storage container create --name <container> --account-name <account>
az storage blob upload --file ./file.pdf --container-name <container> --name <blob>

# App Service
az webapp create --name <app> --resource-group <rg> --plan <plan> --runtime "PYTHON:3.11"
az webapp deploy --name <app> --resource-group <rg> --src-path app.zip
az webapp deployment slot swap --name <app> --resource-group <rg> --slot staging

# AI Services
az cognitiveservices account create --name <name> --resource-group <rg> --kind CognitiveServices --sku S0 --location eastus
az cognitiveservices account keys list --name <name> --resource-group <rg>

# AI Foundry (via AI CLI or Portal - ai.azure.com)
az ml workspace create --name <hub> --resource-group <rg> --kind hub
az ml workspace create --name <project> --resource-group <rg> --kind project --hub-id <hub-id>
```

---

*Happy interviewing! Practice these concepts hands-on using the [Azure Free Account](https://azure.microsoft.com/free/) and the [Azure AI Foundry Portal](https://ai.azure.com).*
