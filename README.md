# ESB CAD Network

The aim of this repository is to link [CSO's Small Area geometries](https://www.cso.ie/en/census/census2016reports/census2016smallareapopulationstatistics/) to [ESB's Transformer Substation locations](https://www.esbnetworks.ie/new-connections/generator-connections/network-capacity-map) via ESB's CAD Network data (`available upon request`).  This will enable grid capacity assessment by region for energy planning & modelling.

This repository assumes the following file structure for the CAD Network file `ESBdata_20200124`:

    ESBdata_20200124
    ├── Ancillary Data
    │   ├── frammelib.rsc
    │   └── mv_index.dgn
    ├── Dig Request Style
    │   ├── HV Data
    │   │   ├── H000000.dgn
    │   │   └── H280000.dgn
    │   └── MV-LV Data
    │       ├── M000040.dgn
    │       ├── M000080.dgn
    │       ├── M040000.dgn
    │       ├── M040040.dgn
    │       ├── M040080.dgn
    │       ├── M040120.dgn
    │       ├── M040200.dgn
    │       ├── M040240.dgn
    │       ├── M040280.dgn
    │       ├── M040320.dgn
    │       ├── M080000.dgn
    │       ├── M080040.dgn
    │       ├── M080080.dgn
    │       ├── M080120.dgn
    │       ├── M080160.dgn
    │       ├── M080200.dgn
    │       ├── M080240.dgn
    │       ├── M080280.dgn
    │       ├── M080320.dgn
    │       ├── M120000.dgn
    │       ├── M120040.dgn
    │       ├── M120080.dgn
    │       ├── M120120.dgn
    │       ├── M120160.dgn
    │       ├── M120200.dgn
    │       ├── M120240.dgn
    │       ├── M120280.dgn
    │       ├── M120320.dgn
    │       ├── M120360.dgn
    │       ├── M160000.dgn
    │       ├── M160040.dgn
    │       ├── M160080.dgn
    │       ├── M160120.dgn
    │       ├── M160160.dgn
    │       ├── M160200.dgn
    │       ├── M160240.dgn
    │       ├── M160280.dgn
    │       ├── M160320.dgn
    │       ├── M160360.dgn
    │       ├── M160400.dgn
    │       ├── M160440.dgn
    │       ├── M200040.dgn
    │       ├── M200080.dgn
    │       ├── M200120.dgn
    │       ├── M200160.dgn
    │       ├── M200200.dgn
    │       ├── M200240.dgn
    │       ├── M200280.dgn
    │       ├── M200320.dgn
    │       ├── M200360.dgn
    │       ├── M200400.dgn
    │       ├── M200440.dgn
    │       ├── M240080.dgn
    │       ├── M240120.dgn
    │       ├── M240160.dgn
    │       ├── M240200.dgn
    │       ├── M240240.dgn
    │       ├── M240280.dgn
    │       ├── M240320.dgn
    │       ├── M240400.dgn
    │       ├── M240440.dgn
    │       ├── M280080.dgn
    │       ├── M280120.dgn
    │       ├── M280160.dgn
    │       ├── M280200.dgn
    │       ├── M280220.dgn
    │       ├── M280240.dgn
    │       ├── M280260.dgn
    │       ├── M280280.dgn
    │       ├── M280320.dgn
    │       ├── M300200.dgn
    │       ├── M300220.dgn
    │       ├── M300224.dgn
    │       ├── M300228.dgn
    │       ├── M300232.dgn
    │       ├── M300236.dgn
    │       ├── M300240.dgn
    │       ├── M300260.dgn
    │       ├── M304220.dgn
    │       ├── M304224.dgn
    │       ├── M304228.dgn
    │       ├── M304232.dgn
    │       ├── M304236.dgn
    │       ├── M308220.dgn
    │       ├── M308224.dgn
    │       ├── M308228.dgn
    │       ├── M308232.dgn
    │       ├── M308236.dgn
    │       ├── M312220.dgn
    │       ├── M312224.dgn
    │       ├── M312228.dgn
    │       ├── M312232.dgn
    │       ├── M312236.dgn
    │       ├── M316220.dgn
    │       ├── M316224.dgn
    │       ├── M316228.dgn
    │       ├── M316232.dgn
    │       ├── M316236.dgn
    │       ├── M320120.dgn
    │       ├── M320160.dgn
    │       ├── M320200.dgn
    │       ├── M320220.dgn
    │       ├── M320240.dgn
    │       └── M320280.dgn
    ├── ESB Networks Data Style Mappings v1.1.pdf
    ├── ESB Networks Data Style Mappings v1.1.xlsx
    ├── ESB Networks Symbol Definitions v1.1.pdf
    ├── ESB Networks Symbol Definitions v1.1.xlsx
    ├── ESBdata_Readme v1.2.pdf
    ├── Safety Information
    │   ├── ESB Letter Dublin.pdf
    │   ├── ESB Letter Regions.pdf
    │   ├── HVIssues.dgn
    │   ├── Safe_System_of_Work_for_Digging_ver0.3.pdf
    │   ├── Safety Stickers.pdf
    │   ├── Transmission Representatives List.doc
    │   ├── avoid_electric_hazards.pdf
    │   ├── esb_networks_avoidances_of_electrical_hazards_when_digging.pdf
    │   └── safe_digging.pdf
    └── System Style
        ├── HV Data
        │   ├── H000000.dgn
        │   └── H280000.dgn
        └── MV-LV Data
            ├── M000040.dgn
            ├── M000080.dgn
            ├── M040000.dgn
            ├── M040040.dgn
            ├── M040080.dgn
            ├── M040120.dgn
            ├── M040200.dgn
            ├── M040240.dgn
            ├── M040280.dgn
            ├── M040320.dgn
            ├── M080000.dgn
            ├── M080040.dgn
            ├── M080080.dgn
            ├── M080120.dgn
            ├── M080160.dgn
            ├── M080200.dgn
            ├── M080240.dgn
            ├── M080280.dgn
            ├── M080320.dgn
            ├── M120000.dgn
            ├── M120040.dgn
            ├── M120080.dgn
            ├── M120120.dgn
            ├── M120160.dgn
            ├── M120200.dgn
            ├── M120240.dgn
            ├── M120280.dgn
            ├── M120320.dgn
            ├── M120360.dgn
            ├── M160000.dgn
            ├── M160040.dgn
            ├── M160080.dgn
            ├── M160120.dgn
            ├── M160160.dgn
            ├── M160200.dgn
            ├── M160240.dgn
            ├── M160280.dgn
            ├── M160320.dgn
            ├── M160360.dgn
            ├── M160400.dgn
            ├── M160440.dgn
            ├── M200040.dgn
            ├── M200080.dgn
            ├── M200120.dgn
            ├── M200160.dgn
            ├── M200200.dgn
            ├── M200240.dgn
            ├── M200280.dgn
            ├── M200320.dgn
            ├── M200360.dgn
            ├── M200400.dgn
            ├── M200440.dgn
            ├── M240080.dgn
            ├── M240120.dgn
            ├── M240160.dgn
            ├── M240200.dgn
            ├── M240240.dgn
            ├── M240280.dgn
            ├── M240320.dgn
            ├── M240400.dgn
            ├── M240440.dgn
            ├── M280080.dgn
            ├── M280120.dgn
            ├── M280160.dgn
            ├── M280200.dgn
            ├── M280220.dgn
            ├── M280240.dgn
            ├── M280260.dgn
            ├── M280280.dgn
            ├── M280320.dgn
            ├── M300200.dgn
            ├── M300220.dgn
            ├── M300224.dgn
            ├── M300228.dgn
            ├── M300232.dgn
            ├── M300236.dgn
            ├── M300240.dgn
            ├── M300260.dgn
            ├── M304220.dgn
            ├── M304224.dgn
            ├── M304228.dgn
            ├── M304232.dgn
            ├── M304236.dgn
            ├── M308220.dgn
            ├── M308224.dgn
            ├── M308228.dgn
            ├── M308232.dgn
            ├── M308236.dgn
            ├── M312220.dgn
            ├── M312224.dgn
            ├── M312228.dgn
            ├── M312232.dgn
            ├── M312236.dgn
            ├── M316220.dgn
            ├── M316224.dgn
            ├── M316228.dgn
            ├── M316232.dgn
            ├── M316236.dgn
            ├── M320120.dgn
            ├── M320160.dgn
            ├── M320200.dgn
            ├── M320220.dgn
            ├── M320240.dgn
            └── M320280.dgn
