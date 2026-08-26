"""
Synthetic safety report generator for OIL India HSSE prototype.
Generates 500 realistic UA/UC, Near-Miss, and Incident reports
covering all 9 IOGP Life-Saving Rule scenarios with SIF/Non-SIF labels.
"""

import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

SITES = [
    "Duliajan Field", "Jorhat Well Pad", "Numaligarh Refinery",
    "Barauni Pipeline", "Dibrugarh Drilling", "Guwahati Terminal",
    "Digboi Pumping Station", "Sibsagar Wellhead",
]

DEPARTMENTS = [
    "Drilling Operations", "Pipeline Maintenance", "Refinery Operations",
    "Wellhead Services", "HSSE Department", "Mechanical Maintenance",
    "Electrical Maintenance", "Civil Engineering",
]

REPORT_TYPES = ["Unsafe Act", "Unsafe Condition", "Near Miss", "Incident"]

# ── SIF-POTENTIAL narratives (tagged by LSR) ──────────────────────────────────
SIF_NARRATIVES = [
    # Energy Isolation
    ("Energy Isolation",
     "Technician began removing flange bolts on a live 6-inch process line without verifying isolation. "
     "Line was still pressurised at 42 bar. Valve isolation had not been completed and LOTO was not applied. "
     "Worker narrowly avoided a pressure release when a colleague intervened. Permit to work was present but isolation steps were skipped."),
    ("Energy Isolation",
     "Electrical technician was working on a 415V motor control panel without lockout tagout. "
     "Panel door was left open with live busbars exposed. A nearby worker accidentally touched an energised conductor "
     "and received a severe electric shock. No isolation certificate was in place."),
    ("Energy Isolation",
     "During scheduled maintenance on a pump, operator started the motor remotely while another worker had hands inside "
     "the rotating assembly. No lockout was applied. Worker's fingers were caught in the impeller."),
    ("Energy Isolation",
     "Pipeline isolation valve was found in open position during a hydro-test operation. "
     "Zero energy verification was not performed. Pressure surged to 80 bar in a supposedly isolated section, "
     "causing a flange gasket to blow out."),

    # Confined Space
    ("Confined Space",
     "Worker entered a crude oil storage tank for inspection without conducting atmospheric testing. "
     "H2S concentration inside was later measured at 45 ppm — well above the IDLH of 10 ppm. "
     "No standby man was present and rescue equipment was not available at the entry point."),
    ("Confined Space",
     "Two workers descended into a wet sump to clear a blockage. No confined space permit was obtained. "
     "Oxygen levels inside measured at 14% — dangerously below the 19.5% minimum. "
     "One worker began feeling dizzy and had to be pulled out."),
    ("Confined Space",
     "Maintenance team entered a heat exchanger shell for tube plugging without completing the atmospheric test. "
     "Entry was made with an invalid permit. Gas monitor alarm was found to be defective."),
    ("Confined Space",
     "Worker entered a confined drain sump to retrieve a tool. No permit was obtained, no standby man assigned, "
     "and no gas test was carried out. Worker lost consciousness due to oxygen deficiency inside."),

    # Hot Work
    ("Hot Work",
     "Welder commenced cutting operations on a pipeline section adjacent to an active flare stack. "
     "No hot work permit was issued for the zone. Gas detector readings were not taken before work began. "
     "A flash fire occurred when escaping gas ignited from the welding sparks."),
    ("Hot Work",
     "Grinding work was performed within 15 metres of an open hydrocarbon drain without a hot work permit. "
     "No fire watcher was assigned. A spark ignited a pool of oil on the ground causing a small fire."),
    ("Hot Work",
     "Welding on a wellhead Christmas tree was started before completing the mandatory gas test. "
     "Flammable vapours were present from a leaking valve. An explosion occurred injuring two workers."),
    ("Hot Work",
     "Oxy-acetylene cutting was carried out in a compressor shed. The fire extinguisher nearby was found empty. "
     "No hot work permit, no gas test, and no fire watch. Shed had poor ventilation."),

    # Line of Fire
    ("Line of Fire",
     "A 20kg tool bag was dropped from a scaffolding platform at 8 metres height while handing over tools. "
     "A worker below was struck on the shoulder despite a hard-hat area being declared. "
     "No tool tethering or toe-boards were installed on the platform."),
    ("Line of Fire",
     "Worker was standing in the line of fire while a hydraulic hose was being pressurised. "
     "The hose connection failed at 120 bar, ejecting the fitting at high velocity past the worker's face. "
     "No exclusion zone had been established."),
    ("Line of Fire",
     "During pig launching from a pipeline, a worker stood directly in front of the launcher barrel. "
     "The pig was ejected unexpectedly due to a valve sequencing error, narrowly missing the worker."),
    ("Line of Fire",
     "A crane wire sling failed while lifting a wellhead valve. The load fell 3 metres and struck a worker "
     "who was standing within the exclusion zone. The sling was past its rated inspection date."),

    # Working at Height
    ("Working at Height",
     "Worker climbed to the top of a 12-metre storage tank to inspect vent valves without wearing a safety harness. "
     "There were no guardrails on the tank roof. Wind speed was 35 km/h. Worker lost footing and was saved by another "
     "worker grabbing his arm."),
    ("Working at Height",
     "Painter was working from a makeshift platform of oil drums stacked 4 metres high without any fall protection. "
     "No scaffold inspection was done. Platform collapsed injuring the worker."),
    ("Working at Height",
     "Technician climbed a 6-metre ladder to access a flare tip without a buddy or fall arrest system. "
     "Ladder was not secured at the top. Worker fell 3 metres when the ladder slipped."),
    ("Working at Height",
     "Workers were removing corrugated roofing sheets on an elevated processing shed. No harnesses, "
     "no edge protection, and no safety net. One worker stepped on a fragile sheet and fell through to the floor below."),

    # Safe Mechanical Lifting
    ("Safe Mechanical Lifting",
     "A 5-tonne BOP was lifted using a crane with a rated capacity of 4 tonnes. No lift plan had been prepared. "
     "The crane wire parted under the overload and the BOP fell onto the wellhead structure."),
    ("Safe Mechanical Lifting",
     "Rigging crew used a damaged wire sling with a visible kink and broken strands to lift a generator set. "
     "The sling failed mid-lift dropping the load onto the crane outrigger. Banksman was not in position."),
    ("Safe Mechanical Lifting",
     "Workers stood beneath a suspended pump casing while the crane operator attempted to manoeuvre it into place. "
     "No exclusion zone was established. The load swung and struck a worker on the hip."),
    ("Safe Mechanical Lifting",
     "Lifting operation was performed without a valid lifting plan or pre-lift inspection of rigging gear. "
     "The shackle pin was found to be unscrewed partway through the lift. Load dropped 2 metres."),

    # Work Authorisation
    ("Work Authorisation",
     "Maintenance crew commenced work on a heat exchanger without a valid permit to work. "
     "The permit had expired the previous day. Line isolation was not verified. "
     "Hot oil at 180°C was released when a flange was broken, burning one worker's hand."),
    ("Work Authorisation",
     "Electrical team started de-cabling a switchgear without a work permit. No JSA was conducted. "
     "An energised cable was cut, causing an arc flash and injuring the electrician's eyes."),
    ("Work Authorisation",
     "Workers broke into a pressurised line without waiting for permit approval. The supervisor had verbally "
     "approved the work but no written permit was raised. An injury resulted from a sudden release of pressure."),
    ("Work Authorisation",
     "Night shift crew continued a job beyond scope of original permit without re-authorization. "
     "Additional isolations needed were not applied. A near-fatal incident occurred from unexpected energy release."),

    # Driving
    ("Driving",
     "Field vehicle was travelling at 85 km/h on a 40 km/h rated access track in heavy rainfall. "
     "Driver was not wearing a seatbelt. The vehicle skidded and rolled over injuring all three occupants."),
    ("Driving",
     "Truck driver was using a mobile phone while driving a heavy goods vehicle on a narrow field road. "
     "Vehicle crossed the centreline and collided head-on with an oncoming field truck."),
    ("Driving",
     "Driver reversed a loaded tanker truck without a banksman in a congested yard. "
     "Reversing camera was defective. The truck reversed into a worker standing behind the vehicle."),
    ("Driving",
     "Worker was driving after a 14-hour shift showing clear signs of fatigue. "
     "Vehicle departed the road and struck a tree. Journey management plan had not been followed."),

    # Bypassing Safety Controls
    ("Bypassing Safety Controls",
     "Process operator bypassed a high-pressure interlock on a separator vessel to avoid a shutdown. "
     "No management of change was completed. Pressure exceeded design limits causing a gasket failure."),
    ("Bypassing Safety Controls",
     "Relief valve on a compressor was found to have been wired shut with lock wire. "
     "Compressor pressure exceeded safety valve set-point by 30%. No authorisation had been obtained."),
    ("Bypassing Safety Controls",
     "Fire and gas detector was found to have been inhibited for 72 hours without a formal bypass permit. "
     "A gas leak occurred during the inhibited period and was not detected for over 2 hours."),
    ("Bypassing Safety Controls",
     "Safety guard on a rotating coupling was removed to allow a running check and not replaced. "
     "Worker's sleeve caught on the exposed shaft causing a degloving injury."),
]

# ── NON-SIF narratives (general observations) ─────────────────────────────────
NON_SIF_NARRATIVES = [
    "PPE not worn correctly — hard hat tilted at angle instead of level. Reminded and corrected on site.",
    "Worker observed not wearing hi-vis vest in a designated PPE zone. Verbal warning issued.",
    "Oil spill of approximately 2 litres observed near pump P-101. Absorbent material deployed and cleaned.",
    "Housekeeping issue in the tool store — tools not returned to designated racks after shift.",
    "Slippery patch of water found near canteen entrance. Wet floor sign placed and area mopped.",
    "Waste segregation not followed — non-hazardous waste disposed in hazardous bin.",
    "Fire exit sign lamp found to be non-functional in Block C. Maintenance work order raised.",
    "Worker observed eating inside the control room — against site hygiene rules. Counselled.",
    "Minor oil stain on walkway near compressor shed. Area barricaded and cleaned.",
    "Emergency eye wash station found blocked by stored materials. Cleared immediately.",
    "Safety notice board in accommodation camp found to have outdated evacuation map. Updated.",
    "First aid box in workshop found to have expired bandages. Restocked.",
    "Drinking water container near wellhead pad found without lid. Replaced with lidded container.",
    "Portable ladder found stored horizontally on the ground instead of secured vertical rack.",
    "Generator fuel cap found missing — temporary cap applied and procurement raised.",
    "Cable drum tripping hazard observed near transformer yard. Cable routed and secured.",
    "Worker spotted without safety glasses while using angle grinder — no active contact with sparks.",
    "Minor paint defect on handrail — does not affect structural integrity. Cosmetic repair scheduled.",
    "Office ergonomics issue — monitor at incorrect height causing neck strain. Adjusted.",
    "Entrance light to pump room not working — torches being used. Light repaired within 4 hours.",
    "Near miss: toolbox slipped from bench and fell on floor — no one injured.",
    "Worker tripped on threshold at workshop entrance. Minor bruising, no lost time. Area marked.",
    "Contractor vehicle observed speeding at 55 km/h in 40 km/h zone. Driver spoken to.",
    "Worn PPE — gloves with minor tear in use. Worker issued replacement gloves.",
    "Drip tray under sampling point full — oil beginning to overflow. Emptied and disposal arranged.",
    "Temporary cable crossing ground-level in pedestrian area without ramp. Cable ramp fitted.",
    "Calibration sticker found expired on pressure gauge PG-214. Instrument sent for calibration.",
    "Waste oil drum stored without bung cap in laydown area. Cap fitted and drum moved to bunded area.",
    "Handrail section loose on staircase to mezzanine floor. Work order raised for repair.",
    "Safety briefing board in drilling contractor camp had wrong emergency contact numbers. Corrected.",
    "Minor leak from pump mechanical seal — drip rate 2 drops/minute. Defect raised for next maintenance window.",
    "Hard hat strap broken on worker entering site — replaced at gate before entry.",
    "Grease found on floor near lathe in maintenance workshop. Cleaned and root cause: lubrication over-applied.",
    "Static discharge grounding strap missing from tanker loading bay. Replacement sourced.",
    "Portable extinguisher inspection tag overdue by 3 days. Inspected and tagged.",
    "Low tyre pressure observed on field vehicle FV-047. Tyre inflated to specification.",
    "Vehicle check-list not signed by driver before departure. Driver reminded of pre-trip inspection requirement.",
    "Stacked materials in laydown area exceeded maximum stacking height. Re-stacked to safe height.",
    "Warning sign at acid storage area faded and illegible. New sign printed and installed.",
    "Worker reported minor cut on index finger from sharp file edge. First aid provided, no LTI.",
]


def generate_dataset(n: int = 500) -> pd.DataFrame:
    records = []
    start_date = datetime(2023, 1, 1)

    # Ensure ~40% SIF-potential (reflecting elevated proportion for demo clarity)
    sif_count    = int(n * 0.40)
    non_sif_count = n - sif_count

    # Build SIF records
    for i in range(sif_count):
        lsr, narrative = random.choice(SIF_NARRATIVES)
        # Add slight variation
        variation = random.choice([
            "", " The area was poorly lit.", " Shift handover had just occurred.",
            " Supervision was absent at the time.", " Worker was new to the site.",
            " The job was being rushed to meet a deadline.",
        ])
        date = start_date + timedelta(days=random.randint(0, 900))
        records.append({
            "report_id":    f"RPT-{2023 + date.year - 2023}-{i+1:04d}",
            "date":         date.strftime("%Y-%m-%d"),
            "site":         random.choice(SITES),
            "department":   random.choice(DEPARTMENTS),
            "report_type":  random.choices(
                REPORT_TYPES, weights=[0.25, 0.20, 0.35, 0.20], k=1)[0],
            "narrative":    narrative + variation,
            "sif_label":    1,    # ground truth (for demo/evaluation only)
            "lsr_ground_truth": lsr,
        })

    # Build Non-SIF records
    for i in range(non_sif_count):
        date = start_date + timedelta(days=random.randint(0, 900))
        records.append({
            "report_id":    f"RPT-{2023 + date.year - 2023}-{sif_count+i+1:04d}",
            "date":         date.strftime("%Y-%m-%d"),
            "site":         random.choice(SITES),
            "department":   random.choice(DEPARTMENTS),
            "report_type":  random.choices(
                REPORT_TYPES, weights=[0.40, 0.35, 0.20, 0.05], k=1)[0],
            "narrative":    random.choice(NON_SIF_NARRATIVES),
            "sif_label":    0,
            "lsr_ground_truth": "None",
        })

    random.shuffle(records)
    df = pd.DataFrame(records)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset(500)
    out_path = "data/sample_reports.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic safety reports -> {out_path}")
    print(f"SIF-potential: {df['sif_label'].sum()} | Non-SIF: {(df['sif_label']==0).sum()}")
