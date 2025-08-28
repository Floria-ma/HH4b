import os
from urllib.parse import urljoin

def download_and_resolve(url: str, filename: str):
    """Download a file and resolve pointer files recursively, keeping the same filename."""
    os.system(f"curl -s -L {url} -o {filename}")

    try:
        with open(filename, 'r') as f:
            lines = f.read().strip().splitlines()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # If file is a pointer (exactly one line and not a 404 error)
    if len(lines) == 1 and "404" not in lines[0].lower():
        pointer = lines[0].strip()

        # Replace the last path component of `url` with the pointer
        base_url = url.rsplit("/", 1)[0] + "/"   # everything up to last slash
        new_url = urljoin(base_url, pointer)

        print(f" -> {filename} is a pointer, resolving {new_url}")
        download_and_resolve(new_url, filename)  # recursive call



def main():
    # Base URLs
    GIT_REPO_JEC = "https://raw.githubusercontent.com/cms-jet/JECDatabase/master/textFiles"
    GIT_REPO_JR = "https://raw.githubusercontent.com/cms-jet/JRDatabase/master/textFiles"
    
    # Configuration
    CORRECTIONS = {
        "data": {
            "2023_runCv4": "Summer23Prompt23_RunCv4_V3_DATA",
            "2023_runCv123": "Summer23Prompt23_RunCv123_V3_DATA",
            "2023BPix_runD": "Summer23BPixPrompt23_RunD_V3_DATA"
        },
        "MC": {
            "2023mc": "Summer23Prompt23_V3_MC",
            "2023BPixmc": "Summer23BPixPrompt23_V3_MC"
        }
    }

    # File templates
    FILE_TEMPLATES = {
        "AK8": {
            "MC": [
                "L1FastJet_AK8PFPuppi.txt",
                "L2Relative_AK8PFPuppi.txt",
                "UncertaintySources_AK8PFPuppi.txt",
                "Uncertainty_AK8PFPuppi.txt"
            ],
            "data": [
                "L2Relative_AK8PFPuppi.txt",
                "L2L3Residual_AK8PFPuppi.txt"
            ]
        },
        "AK4": {
            "MC": [
                "L1FastJet_AK4PFPuppi.txt",
                "L2Relative_AK4PFPuppi.txt",
                "UncertaintySources_AK4PFPuppi.txt",
                "Uncertainty_AK4PFPuppi.txt"
            ],
            "data": [
                "L2Relative_AK4PFPuppi.txt",
                "L2L3Residual_AK4PFPuppi.txt"
            ]
        }
    }

    # JR/JRC files
    JR_FILES = [
        "Summer23Prompt23_RunCv1234_JRV1_MC/Summer23Prompt23_RunCv1234_JRV1_MC_PtResolution_AK8PFPuppi.txt",
        "Summer23Prompt23_RunCv1234_JRV1_MC/Summer23Prompt23_RunCv1234_JRV1_MC_SF_AK8PFPuppi.txt",
        "Summer23Prompt23_RunCv1234_JRV1_MC/Summer23Prompt23_RunCv1234_JRV1_MC_PtResolution_AK4PFPuppi.txt",
        "Summer23Prompt23_RunCv1234_JRV1_MC/Summer23Prompt23_RunCv1234_JRV1_MC_SF_AK4PFPuppi.txt",

        "Summer23BPixPrompt23_RunD_JRV1_MC/Summer23BPixPrompt23_RunD_JRV1_MC_PtResolution_AK8PFPuppi.txt",
        "Summer23BPixPrompt23_RunD_JRV1_MC/Summer23BPixPrompt23_RunD_JRV1_MC_SF_AK8PFPuppi.txt",
        "Summer23BPixPrompt23_RunD_JRV1_MC/Summer23BPixPrompt23_RunD_JRV1_MC_PtResolution_AK4PFPuppi.txt",
        "Summer23BPixPrompt23_RunD_JRV1_MC/Summer23BPixPrompt23_RunD_JRV1_MC_SF_AK4PFPuppi.txt",
    ]

    def build_urls(correction_type: str, dataset: str) -> [str]:
        """Build URLs for JEC files for a given dataset and type."""
        correction_name = CORRECTIONS[correction_type][dataset]
        files = FILE_TEMPLATES["AK8"][correction_type] + FILE_TEMPLATES["AK4"][correction_type]
        return [f"{GIT_REPO_JEC}/{correction_name}/{correction_name}_{file}" for file in files]

    def build_jr_urls() -> [str]:
        """Build URLs for JR files."""
        return [f"{GIT_REPO_JR}/{file}" for file in JR_FILES]

    # Generate all URLs
    gitpaths = {
        "data": {
            dataset: build_urls("data", dataset) for dataset in CORRECTIONS["data"]
        },
        "MC": {
            dataset: build_urls("MC", dataset) for dataset in CORRECTIONS["MC"]
        }
    }

    jr_urls = build_jr_urls()

    # Download files
    for category, datasets in gitpaths.items():
        for dataset, urls in datasets.items():
            print(f"\nDownloading {category}/{dataset} files")
            for url in urls + jr_urls:
                filename = os.path.basename(url)
                print(f"Downloading: {filename}")
                download_and_resolve(url, filename)

    def build_jec_inputs(): 
        filelist = [file for file in os.listdir() if os.path.isfile(file)]

        coffea_tags = {
            "Uncertainty": "junc",
            "L1FastJet": "jec",
            "L2Relative": "jec",
            "L2L3Residual": "jec",
            "SF": "jersf",
            "PtResolution": "jr",
        }

        newfilelist = []
        for file in filelist:
            base, ext = os.path.splitext(file)  # ("UncertaintySources_AK8PFPuppi", ".txt")
            for key, tag in coffea_tags.items():
                if key in base:  # look only at the base
                    new_name = f"{base}.{tag}{ext}"
                    os.rename(file, new_name)
                    print(f"Renamed {file} → {new_name}")
                    newfilelist.append(new_name)
                    break
        
        filelist = newfilelist # lazy man's fix

        MC_files = [file for file in filelist if "MC" in file] 
        DATA_files = [file for file in filelist if "DATA" in file] 

        data_out = {
            "AK4": {

            },
            "AK8": {
                
            }
        } 
        mc_out = {
            "AK4": {

            },
            "AK8": {
                
            }
        }

        for key, value in CORRECTIONS["data"].items(): 
            data_out["AK4"][key] = [file for file in DATA_files if value[:-6] in file and "AK4" in file] # :-6 on value removes up to and including the # in V#_DATA
            data_out["AK8"][key] = [file for file in DATA_files if value[:-6] in file and "AK8" in file]

        for key, value in CORRECTIONS["MC"].items(): 
            mc_out["AK4"][key] = [file for file in MC_files if value[:-4] in file and "AK4" in file] # :-4 on value removes up to and including the # in V#_MC
            mc_out["AK8"][key] = [file for file in MC_files if value[:-4] in file and "AK8" in file] 



        mc_out["AK4"]["2023mc"] += [file for file in MC_files if "Summer23Prompt23RunCv1234_JRV1_MC" in file and "AK4" in file]
        mc_out["AK8"]["2023mc"] += [file for file in MC_files if "Summer23Prompt23RunCv1234_JRV1_MC" in file and "AK8" in file]

        mc_out["AK4"]["2023BPixmc"] += [file for file in MC_files if "Summer23BPixPrompt23_RunD_JRV1_MC" in file and "AK4" in file]
        mc_out["AK8"]["2023BPixmc"] += [file for file in MC_files if "Summer23BPixPrompt23_RunD_JRV1_MC" in file and "AK8" in file]


        return data_out, mc_out
    
    data_out, mc_out = build_jec_inputs() 
    print(mc_out) 
    print(data_out)
if __name__ == "__main__":
    main()
