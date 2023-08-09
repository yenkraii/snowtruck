import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle
import xgboost as xgb
from PIL import Image
from sklearn import preprocessing

#st.set_page_config(page_title="SnowTruck", page_icon=":truck:")

#st.image(snowtruck_logo)

snowtruck_logo = Image.open("images/Snowtruck_ProductBox.jpeg")

st.image(snowtruck_logo, width=700)
st.title("SnowTruck:minibus:")

# connects to the snowflake account 
# if need to use the data there
conn = snowflake.connector.connect(**st.secrets["snowflake"])

# tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Predicting Daily Sales","Profit Prediction","MBA + Uplift Modelling","tab4","tab5"])

with tab1:
  import xgboost as xgb
  
# Define the app title and favicon
  st.title('How much can you make from the TastyBytes locations? 💡')
  st.markdown("**Tell us more on what you want to predict!**")
  
  # Loading the pickle & dataset
  # with open('xgb_alethea.pkl', 'rb') as file:
  #       xgb_alethea = pickle.load(file)
  # df = pd.read_csv('df_aletheaDOW.csv')

  wd_mapping  = { 'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6 }
  wd_reverse_mapping = {v: k for k, v in wd_mapping.items()}
  wd_labels = [wd_reverse_mapping[i] for i in sorted(wd_reverse_mapping.keys())]

  bn_mapping = { "Cheeky Greek": 0,
                  "Guac n' Roll": 1,
                  "Smoky BBQ": 2,
                  "Peking Truck": 3,
                  "Tasty Tibs": 4,
                  "Better Off Bread": 5,
                  "The Mega Melt": 6,
                  "Le Coin des Crêpes": 7,
                  "The Mac Shack": 8,
                  "Nani's Kitchen": 9,
                  "Plant Palace": 10,
                  "Kitakata Ramen Bar": 11,
                  "Amped Up Franks": 12,
                  "Freezing Point": 13,
                  "Revenge of the Curds": 14 }
  bn_reverse_mapping = {v: k for k, v in bn_mapping.items()}
  bn_labels = list(bn_mapping.keys())

  ct_mapping = { 'San Mateo': 0, 'Seattle': 1, 'New York City': 2, 'Boston': 3, 'Denver':4 }
  ct_reverse_mapping = {v: k for k, v in ct_mapping.items()}
  ct_labels = list(ct_mapping.keys())

  tl_mapping = { 'Veterans Park': 0, 'City of New York': 1, 'Clason Point Park': 2, 'Stanley Bellevue Park': 3, "Hunt's Point Farmers Market": 4, 'Musee Fogg': 5,
                  'Rainey Park': 6, 'Tiffany Street Pier': 7, "Jeff Koons' Balloon Flower Sculpture at WTC": 8, 'Museum Of Modern Art': 9, 'Garland Street Park': 10,
                  'Crestmoor Park': 11, 'Johnson Habitat Park': 12, 'Westcrest Park': 13, 'Fremont Peak Park': 14, 'Christodora House': 15, 'ArchCare at Mary Manning Walsh Home': 16,
                  'West Magnolia Playfield': 17, 'All About Seniors': 18, 'Southwest Auto Park': 19, 'Haven of Care Assisted Living': 20, 'Ellis': 21, 'Garland David T Park': 22,
                  'Ashley': 23, 'Colorado Tennis Association': 24, 'Ryan Play Area': 25, 'Umana Schoolyard': 26, 'Say Yes To Education Teachers College': 27, 'Harlem River Park': 28, 'City Of Kunming Park': 29,
                  'Avalanche Ice Girls': 30, 'Red Light Camera 6th & Lincoln': 31, 'Operation School Bell': 32, 'Denver Art Museum': 33, 'Crating Technologies': 34, 'Frances Wisebart Jacobs Park': 35,
                  'Douglass Fredrick Park': 36, 'Granary Burying Ground': 37, 'Ez Pass': 38, 'Lo Presti Park': 39, 'Franco Bernabe Indio Park': 40, 'Upholstery Outfitters of Seattle': 41, 'Marine Park': 42,
                  'Silver Lake Park': 43, 'Historic New England': 44, 'Maison Blanche': 45, 'Northwest Blind Repair': 46, 'Faneuil Square': 47, 'Felix Potin Chocolatier': 48,
                  'Bell Street Park Boulevard': 49, 'Medcom': 50, 'Washington Park': 51, 'InnovAge Headquarters': 52, 'Eat Harlem': 53, 'Weider Park': 54, 'It Professional': 55, 'Ambassador Candy Store': 56,
                  'Emeritus at West Seattle': 57, 'Army National Guard Recruiting': 58, 'Evergreen Park': 59, 'St John Vianney Theological Seminary': 60, 'Education Commission of the States': 61,
                  'Siemann Educational': 62, 'Sanderson Gulch Irving and Java': 63, "Lower Allston's Christian Herter Park": 64, 'Conley School Play Yard': 65, 'Colorado Rockies Baseball Club': 66,
                  'City Of Cuernavaca Park': 67, 'Father Duffy Square': 68, 'Wyman Multiple Path Iii': 69, 'Dayspring Villa': 70, 'Morrison George Sr Park': 71, 'Eisenhower Mamie Doud Park': 72,
                  'A Place for Mom': 73, 'Cal Anderson Park': 74, 'Woodland Park Off Leash Area': 75, 'Blitz Bowl Football Bowling': 76, 'Steele Street Park': 77, 'American Museum Of Western Art': 78,
                  'Museum Of Contemporary Art Denver': 79, 'Swansea Park': 80, 'East Portal Viewpoint': 81, 'Dudley Town Common': 82, 'Meadowbrook Playfield': 83, 'Spokane Street Bridge': 84,
                  'Allen Park': 85, 'Hungarian Monument': 86, 'Nutbox': 87, 'Molly Brown Summer House': 88, 'Horiuchi Park': 89, 'Museum Of Comic And Cartoon Art': 90, 'McIntyre Organ Repair Service': 91,
                  'Korean War Veterans Memorial Park': 92, 'Wai Wah Market': 93, 'Amasia College': 94, 'Arborway Overpass Path': 95, 'Damrosch Park': 96, 'Riverside Park South': 97,
                  'Gopher Broke Farm': 98, 'Harlem Carmiceria Hispana & Deli': 99, 'Gapstow Bridge': 100, 'Queens Farm Park': 101, 'Spring Manor': 102, 'Evolving Vox': 103, 'TD Garden': 104,
                  'Fourth Street Park': 105, 'Weatherflow Dba Wind Surf': 106, 'Olmsted Green': 107, 'Central Park Tennis Center': 108, 'Uplands Park': 109, "Chef's Cut": 110, 'Me Kwa Mooks Park': 111,
                  'Dunbarton Woods': 112, 'Hirshorn Park': 113, 'Lightrail Plaza': 114, 'Cherry Creek Gun Club': 115, 'Spectrum Retirement Communities': 116, 'Mass Historical Soc': 117,
                  "Soldiers' & Sailors' Monument": 118, 'Conservancy For Historic Battery Park': 119, 'Union Square Metronome': 120, 'Espresso Repair Experts': 121, 'View Ridge Es Playfield': 122,
                  'Bedford Sub Zero Appliance Repair': 123, 'Northlake Park': 124, 'Hing Hay Park': 125, 'Boston University Bridge': 126, 'Amedei': 127, 'Myrtle Edwards Park': 128, 'Pinnacle City Park Hoa': 129,
                  'Friends of Historic Ft Logan': 130, 'Pulaski Park': 131, 'Greenway Park': 132, 'Futura Food': 133, 'Korean War Veterans Memorial': 134, 'Umass Harborwalk': 135, 'Nathan Hale Stadium Complex': 136,
                  'Are East River Science Park': 137, 'Langone Park': 138, 'Atlantic City Boat Ramp': 139, 'Ny Newstand Candy and Grocery': 140, 'American University of Barbados School of Medicine': 141,
                  'Cakes Confidential': 142, 'TeenLife Media': 143, 'Bradner Gardens Park': 144, 'The Jewish Museum': 145, 'Berklee College Of Music': 146, 'Rockledge Street Urban Wild': 147,
                  'Arsenal North': 148, 'Laurelhurst Beach Club': 149, 'Chinatown Fair Family Fun Center': 150, 'Seaport Common': 151, 'Cuny City College': 152, 'Bar And Grill Park': 153,
                  'Seattle Interactive Media Museum': 154, 'Symphony Road Garden': 155, 'Robert H McWilliams Park': 156, 'Museum Of American Humor': 157, 'PasquinelS Landing': 158, 'Joe Moakley Park': 159,
                  'Lower Woodland Park Playfields': 160, 'Place': 161, "Eidem's Custom Upholstery": 162, 'West Central Grounds Maint': 163, 'Merrill Gardens at Queen Anne': 164, 'C & B Candy Store': 165,
                  'Casse Cou': 166, 'Vernam Barbadoes Peninsula': 167, 'Connections': 168, 'Bridge Gear Park': 169, 'Old Harbor Park': 170, 'Moynihan Spray Deck': 171,
                  "Hillel Foundation of Greater Boston B'nai B'rith": 172, 'Grindline': 173, 'Boston University School of Public Health': 174, 'Aquarium New England': 175, 'City Square Park': 176,
                  'Boston University': 177, 'Denison': 178, 'Chinook Beach Park': 179, 'Jack Block Park': 180, 'TrollS Knoll': 181, 'Architectural Heritage Foundation': 182, 'NP Agency Inc': 183, 'Drummers Grove': 184,
                  'Cinderella Tailors': 185, 'Pilsudski Josef Inst of Amer': 186, 'Village Community Boathouse': 187, 'Wrench Bicycle Workshop': 188, 'Greenwood Park': 189, 'Bodyslimdown Garcinialossfat': 190,
                  'Drug Treatment Centers Manhattan': 191, 'Carson Beach': 192, 'Anatolia College': 193, 'Boston History Center And Museum': 194, 'Arctic Viking Experts': 195, 'York Playground': 196, 
                  'United Machine Shop Service': 197, 'Rainier Beach Urban Farm and Wl': 198, 'Share The Care': 199, 'Guan Hua Candy Store': 200, 'Color Factory': 201, "Christie's Sculpture Garden": 202,
                  'Sugarfina': 203, 'Halal Food': 204, 'Matthes Roland': 205, 'Citarella Gourmet Market Upper West Side': 206, 'Beethoven School Play Area': 207, 'African Meeting House': 208,
                  'Mansfield Street Dog Park': 209, 'Darren Sub Zero Repair Master': 210, 'Powell Barnett Park': 211, 'Spada Homes': 212, 'Commonwealth Avenue Outbound': 213, 'Smith Park Pumptrack': 214,
                  'OReilly Way Court': 215, 'Vietnam War Memorial': 216, 'Senior Guidance': 217, 'Knapp': 218, 'Lindsley Henry S Park': 219, 'Magness Arena': 220, 'Mcmeen': 221, 'City Of Takayama Park': 222,
                  'Cardinal Cushing Memorial Park': 223, 'West Street': 224, 'Kendall and Lenox Streets Garden': 225, 'Boston Design Center Plaza': 226, 'Suffolk University Law School': 227, 'Pro Arts Consortium': 228,
                  'University College': 229, 'Denver Drug Rehab': 230, 'Seattle Parks and Recreation': 231, 'University of Washington Virology Research Clinic': 232, 'Forney Museum': 233, 'Doull': 234,
                  'Tzun Tzu Military History Museum': 235, 'Northside Park': 236, 'Ercolini Park': 237, 'Arbor Heights Elder Care': 238, 'Forest Hills Preserve': 239, 'Victory Road Park': 240, 'Schmitz Boulevard': 241,
                  'Kirke Park': 242, 'Suboxone Treatment Clinic': 243, 'Fearless Girl': 244, 'Grant Ranch Village Center': 245, 'Fairview': 246, 'South Slope Dog Run': 247, 'Seward Park': 248, 'Porzio Park': 249,
                  'Union Square Plaza': 250, 'Tent City Courtyards': 251, 'Bayswater Street': 252, 'Prezant Sonia': 253, 'Benjamin Franklin Institute Of Technology': 254, 'Samuel Adams Statue': 255,
                  'General Grant National Memorial': 256, 'High Rated Repair': 257, 'NW Caravan Treasures': 258, 'Presbyterian Retirement Communities Northwest': 259, 'Fuller Theological Seminary': 260,
                  'Harriet Tubman Square': 261, 'Chelsea Creek': 262, 'E & L Seafood': 263, 'Schenck': 264, "See's Candies": 265, 'Harvey Park': 266, 'Laderach': 267, 'Arthur Ashe Stadium': 268,
                  'Dr Blanche Lavizzo Park': 269, 'Florence of Seattle': 270, 'I 90 Interchange': 271, 'Leo M Birmingham Parkway': 272, "Green Lake Pitch n' Putt": 273, 'Homewood Park': 274, 'Miller Triangle': 275,
                  'Leonard P Zakim Bunker Hill Memorial Bridge': 276, 'Rose Fitzgerald Kennedy Greenway': 277, 'Bryant Webster': 278, 'Exceptional Voice': 279, 'Senior Care Authority': 280, 'Millennium Bridge': 281,
                  'Dreamland Wax Museum': 282, 'Simmons College': 283, 'Rogers Park': 284, "Jack O' Lantern Journey": 285, 'Harvard University Harvard Medical School': 286, 'Massport Harborwalk': 287,
                  'South Boston Bark Park': 288, 'Williams Street Iii': 289, 'MassArt Art Museum': 290, 'Carniceria La Rosa': 291, 'Colorado Home Care': 292, 'High Line East': 293, 'Invisible Museum': 294,
                  'Crowley Boats': 295, 'Seattle Tacoma Appliance Repair': 296, 'Burke Gilman Trail': 297, 'ABC Appliance Service': 298, 'Bow Bridge': 299, 'Rider Oasis': 300, 'Drug Rehab and Sober Living Seattle': 301,
                  'The Seattle Great Wheel': 302, 'ARS Appliance Repair Service': 303, 'Columbia Road Day Boulevard': 304, 'Clarendon Street Totlot': 305, "Jim's Cobbler Shoppe": 306, 'Montlake Playfield': 307,
                  'National Oceanic and Atmospheric': 308, 'Met Fresh': 309, 'Carleton Highlands': 310, 'Meril Gardens': 311, 'Halal Cart': 312, 'Emerson College': 313, 'Boston Museum': 314, 'Shi Eurasia': 315,
                  'Pasta Resources': 316, 'McPhilomy Commercial Products': 317, 'Aspen University': 318, 'Thomas J Cuite Park': 319, 'Carson': 320, 'Pulaski Park and Playground': 321,
                  'The Langland House Adult Family Home': 322, 'Refrigerator Repair': 323, 'Hilltop Appliance Repair': 324, 'Japan Premium Beef': 325, 'Columbus Park': 326, 'Malaysia Beef Jerky': 327,
                  'Pinecrest Village Park': 328, 'Montefiore Square': 329, 'Jacques Torres Chocolate': 330, 'Webster Park': 331, 'The Laser Dome': 332, 'American Freestyle Alterations': 333, 'Gorman Park': 334,
                  'John Copley Statue': 335, 'Dennis Street Park': 336, 'Museum Of Afro American History': 337, 'Boston University Sargent Choice Nutrition Center': 338, 'Mt Bowdoin Green': 339,
                  'Martinez Joseph P Park': 340, 'Reclining Liberty': 341, 'Twenty Four Sycamores Park': 342, 'Underground Paranormal Experience': 343, 'E C Hughes Playground': 344, 'Bitter Lake Playfield': 345,
                  'American Cultural Exchange': 346, 'Era Living': 347, 'Jefferson Park': 348, 'Macarthur Park': 349, 'Li Lac Chocolates': 350, 'OU Israel Free Spirit': 351,
                  'Society For the Preservation of New England Antqts': 352, 'Commonwealth Pier': 353, 'Judkins Park P Patch': 354, 'Hyperspace': 355, "Harry & Ida's Meat and Supply": 356,
                  'Home Espresso Repair and Cafe': 357, 'Adam Tailoring & Alterations': 358, 'Spruce Street Mini Park': 359, 'Chinatown Park': 360, "Alioto's Garage": 361, 'Tracey Lactation': 362,
                  'Execupark Incorporated': 363, 'San Mateo County Times': 364, 'SSB Kitchen': 365, 'Owens Electric & Solar': 366, 'VIP Petcare': 367, 'Magic Mountain Playground': 368, 'Health Street': 369,
                  'Paul Revere Mall': 370, 'Blake Estates Urban Wild': 371, 'John Hancock Tower': 372, 'Long Wharf Boat Access': 373, 'Pioneer Square': 374, 'Demonstration Garden': 375, 'Mount Baker Park': 376,
                  'Massachusetts Democratic Party': 377, 'Bremen Street Dog Park': 378, 'Sugar and Plumm': 379, 'Arc Watch Works & Engraving': 380, 'Chocolate Covered Everything': 381,
                  'Simmons College Residence Campus': 382, 'Paul Revere Park': 383, 'The Highlands': 384, 'Ace Hardware': 385, 'Apartments at 225 Catalpa St': 386, 'Macadamian': 387, 'Tina Canada Nail Technician': 388,
                  'Shore Vu Laundromat': 389, 'Faenzi Associates': 390, 'Donna Ornitz': 391, 'Bridgepointe': 392, 'Nordstrom': 393, 'Mark Woods BPIA Insurance': 394, 'KG Fitness Studio': 395, 'West Elm': 396,
                  'Snap Fitness': 397, 'Swarovski': 398, 'King Pang Hairdressing': 399, 'Bay Area Sunrooms': 400, 'Sasi Salon': 401, 'Michaels Stores': 402, 'Apartments at 321 Monte Diablo Ave': 403,
                  'Apartments at 20 Hobart Ave': 404, 'Apartments at 602 Cypress Ave': 405, 'Mason James K Insurance': 406, 'Mcc': 407, 'GreenCal Solar': 408, 'Marco Nascimento Brazilian Jiu Jitsu': 409,
                  'Precision Concrete Cutting': 410, 'Townhouses at 821 Laurel Ave': 411, 'Toi Lynn Wyle MS MFT ERYT': 412, 'Best Western Coyote Point Inn': 413, 'California Bank Trust': 414, 'Reflektion': 415,
                  'Eclipz Hair Designs': 416, 'Park Vanderbilt': 417, 'DP Systems': 418, 'Q Fix PC Services': 419, 'Apartments at 211 E Santa Inez Ave': 420, 'Associated Psychological Services': 421, 'Brust Park': 422,
                  'Chanin News Corporation': 423, 'Kernan Farms': 424, 'Seattle Tower': 425, 'East Queen Anne Playground': 426, 'Matthews Beach': 427, 'Open Water Park': 428, 'Waterfall Garden': 429,
                  'Bar S Playground': 430, 'Bellevue Place': 431, "Expeditors Int'l of Washington": 432, 'Charlestown Navy Yard': 433, 'Fire Alarm House Grounds': 434, 'Nonquit Green': 435, 'Design Museum Boston': 436,
                  'Ideal Health Clinic': 437, 'Hi View Apartments': 438, 'LifeMoves First Step for Families': 439, 'Aikido By The Bay': 440, 'St Timothy School': 441, 'Apartments at 1116 1120 Folkstone Ave': 442,
                  'U S First Federal Credit Union': 443, 'Apartments at 3149 Casa De Campo': 444, 'Reservation Road Park': 445, 'Maximum Performance Hydraulics': 446, 'Bitter Lake Open Space Park': 447,
                  'Conley and Tenean Streets Park': 448, 'My Paper Pros': 449, 'Gourmet Today Publctn': 450, 'Stars Clinic': 451, "Muslim Children's Garden": 452, 'San Mateo DMV Office': 453,
                  'San Bruno Dog Obedience School': 454, 'Barastone': 455, 'First Priority Financial': 456, 'Reposturing The Pain Elimination Method': 457, 'Apartments at 603 E 5th Ave': 458,
                  'East 19th Avenue Apartments': 459, 'Diamond Motors': 460, 'M Street Beach': 461, 'Puddingstone Park': 462, 'Hernandez Schoolyard': 463, 'Machate Circle': 464, 'Unigo': 465,
                  'Judkins Park And Playfield': 466, 'Magnolia Greenbelt': 467, 'Louis Valentino Jr Park & Pier': 468, 'Scott Eaton Supreme Lending': 469, 'Jolene Whitley Hair Design & Replacement': 470,
                  'RePlanet Recycling': 471, 'American Prime Financial': 472, "Bo Jonsson's Foreign Car Service": 473, 'Apex Iron': 474, 'Glass Express': 475, 'Financial Title Company': 476, 'Golden 1 Credit Union': 477,
                  'Esthetique European Skin Care Clinic': 478, 'Y2K Nails': 479, 'Gazelle Developmental School': 480, 'Apartments at 731 N Amphlett Blvd': 481, 'Empowerly': 482, 'Williams-Sonoma': 483,
                  'Brooklyn Bridge Park Pier 5': 484, 'Ryan Playground': 485, 'Hayes Park': 486, 'Pine Street Park': 487, 'Woodhaven': 488, 'Pulmonary Wellness & Rehabilitation Center': 489, 'UCSF Medical Center': 490,
                  'Bologna Chiropractic & Sports Care': 491, 'Apartments at 217 Villa Ter': 492, 'Standard Parking': 493, 'Peninsula Golf & Country Club': 494, 'Thai Art of Massage': 495, 'PattenS Cove': 496,
                  'Lewis Mall': 497, 'American Legion Highway': 498, 'Sustainable Table': 499, 'Lulo Restaurant': 500, 'Cronin Playground': 501, 'Muse Beverly Lmft': 502, 'Release From Within': 503,
                  'Wells Fargo ATM': 504, 'K & W Liquors': 505, 'Jibe Promotional Marketing': 506, 'Meetinghouse Hill Overlook': 507, "Jeffery's Dog & Cat Grooming": 508, 'Come C Interiors': 509,
                  'Apartments at 816 Highland Ave': 510, 'Yok Thai Massage': 511, "Seattle Children's Museum": 512, 'Downer Dog Park': 513, 'San Mateo Auto Works': 514, 'Jazz Museum': 515,
                  'Stony Brook Reservation Iii': 516, 'Eurasian Auto Repair': 517, "DiMenna Children's History Museum": 518, 'Asian Street Eats': 519, 'Leonidas Fresh Belgian Chocolates': 520, 'South Beach': 521,
                  'Park Crest': 522, 'Nathan Hale Playfield': 523, 'Simply Frames & Miner Gallery': 524, 'Aquarist World': 525, "Women's Rights Pioneers Monument": 526, 'Givans Creek Woods': 527,
                  'Our Lady Queen Of Angels': 528, 'National Museum Of Mali': 529, 'Babi Yar Park': 530, 'Eastern Star Masonic Retirement Community': 531, 'Wooden Open': 532, 'Village Greens Park': 533,
                  'Prometheus Statue': 534, 'Chelsea Eats': 535, 'Amersfort Park': 536, 'Coney Island Beach and Boardwalk': 537, 'Underwood Park': 538, 'Kaboom Virtual Reality Arcade': 539,
                  'Troy Chavez Memorial Peace Garden': 540, 'Coney Island Beach': 541, 'Columbus Gourmet Food': 542, 'Core Social Justice Cannabis Museum': 543, 'Elc Playlot': 544, 'Armenian Heritage Park': 545,
                  'Meow Wolf Denver Convergence Station': 546, 'Department of Psychiatry of Columbia University': 547, "Bill's Lobby Stand": 548, 'Maple Grove Park': 549, 'Frederick Douglass Park': 550, 
                  'Farmers Insurance Group': 551, 'Bay Motors': 552, 'Laurelwood Shopping Center': 553, 'Lily European Pedi Mani Spa': 554, 'Star Smog San Mateo': 555, 'Mattahunt School Entrance Plaza': 556,
                  'Association of Independent Colleges & Universities of Mass': 557, 'University Of Denver': 558, 'Ball arena formerly Pepsi Center': 559, 'Salmagundi Club Museum': 560, 'Vesey St Edible': 561,
                  'Vintage Motos Museum': 562, 'The Great Lawn Park': 563, 'Veterans Field': 564, "Sloan's Lake Park": 565, 'The Tiger Lily Kitchen': 566, 'Alcoholics Anonymous': 567,
                  'Penniman Road Play Area': 568, 'Vfw Parkway': 569, 'Jimi Hendrix Statue': 570, 'I S 049 Bertha A Dreyfus': 571, 'Black Top Street Hockey': 572, 'Yue Fung Usa Enterprise Incorporated': 573,
                  'Thornton Creek Natural Area': 574, 'Schmitz Preserve Park': 575, 'Seattle Sub Zero Repair': 576, 'Denver Museum Of Nature & Science': 577, 'Samuels': 578, 'Fishback Park': 579,
                  'Bryant Hill Garden': 580, 'Addiction Recovery Hotline Manhattan': 581, 'Rockaway Beach And Boardwalk': 582, 'New York Career Training & Advancement': 583, 'Tottenville Pool': 584,
                  'Gateway Newstands': 585, 'Daniel Carter Beard Mall': 586, 'Hallack Park': 587, 'Silverman Melvin F Park': 588, 'Balfour': 589, 'Russell Fire Club': 590, 'Central Vacuum Service': 591,
                  'Pacific College of Allied Health': 592, 'CommonGround Golf Course': 593, 'Westerleigh Park': 594, 'City Of Ulaanbaatar Park': 595, 'Cycleton Lowry': 596, 'West 14th Candy Store': 597,
                  'Oldest Manhole Cover In NYC': 598, 'Waldorf Schls Fund': 599, 'Invictus Care': 600, 'Seton Falls Park': 601, 'Lewis Mall Harborpark': 602, 'Wheelock College of Education & Human Development': 603,
                  'InnovAge PACE Denver': 604, "Weiskind's Appliance": 605, 'Huguenot Ponds Park': 606, 'Bayaud Park': 607, 'South Street Courts': 608, 'Lowry Open Space': 609, 'Dailey Park': 610,
                  "University Of Denver Pioneers Men's Lacrosse": 611, 'Parker Terrace': 612, 'Seattle Medical & Rehabilitation Center': 613, 'Muscota Marsh': 614, 'Midtown Catch': 615, 'Macombs Dam Park': 616,
                  'Swansea Neighborhood Park': 617, 'Hampden Heights Park': 618, 'Case Gym': 619, 'Maple Sonoma Streets Community Park': 620, 'Lopresti Park': 621, 'Boston Nature Center Visitor Ctr': 622,
                  "The Vilna Shul Boston's Center for Jewish Culture": 623, 'The ED Clinic': 624, 'F 15 Park': 625, 'Boyden Park': 626, 'Columbia University Medical Center Medical School': 627,
                  'The General Theological Seminary': 628, 'Blood Manor': 629, 'Prentis I Frazier Park': 630, 'O Sullivan Art Museum': 631, 'Dry Gulch Trail': 632, "Martha's Garden": 633, 'Garfield Lake Park': 634,
                  'Sanchez Paco Park': 635, 'Shadow Array': 636, 'Arthur Brisbane Memorial': 637, 'Alford Street Bridge': 638, 'Mgh Institute Of Health Professions': 639, 'Norman B Leventhal Park': 640,
                  'They Shall Walk Museum': 641, 'Marshall Park': 642, 'Lowry': 643, 'Inspir Carnegie Hill': 644, 'Oakland': 645, 'Carla Madison Dog Park': 646, 'Magna Carta Park': 647,
                  'International Neural Renewal': 648, 'Wallace Park': 649, 'Garden Place': 650, 'Staats Circle': 651, 'Sugar Town': 652, 'RTV Exotics': 653, 'Worldwide Adult DayCare': 654,
                  'Travel Center Specialists': 655, 'Rosamond Park': 656, 'Sobriety House': 657, 'Envision Response': 658, 'Ruby Hill': 659, 'Mcclain Thomas Ernest Park': 660, 'Harlem Chocolate Factory': 661,
                  'Gruppo Cioccolato Internazionale': 662, 'L J Campbell Walter': 663, 'Nira Rock': 664, 'Long Lane Meeting House': 665, 'Dreiling Ruth Lucille Park': 666, 'Trinity Park': 667, "Dylan's Candy Bar": 668,
                  'Baker Chocolate Factory': 669, 'Rolling Bridge Park': 670, 'Central Seattle Panel of Consultants Inc': 671, 'Lake City Mini Park': 672, 'Licorice Fern Na On Tc': 673, 'Nespresso': 674,
                  'Peoples Computer Museum': 675, 'Superb Custom Tailors': 676, 'Dacia Woodcliff Community Garden': 677, 'Stony Brook Commons Park': 678, 'Pier Sixty New York City Event and Weddings Venue': 679,
                  'AXA Equitable Center': 680, 'University Park': 681, 'Healing Lifes Pains': 682, 'Noras Woods': 683, 'The Summit At First Hill': 684, "Boston University Women's Council": 685,
                  'Pendleton Miller Playfield': 686, 'UW Medicine Diabetes Institute': 687, 'Woodland Park': 688, 'Key To Amaze': 689, 'Vladeck Park': 690, 'German House': 691, 'Hope Sculpture By Robert Indiana': 692,
                  'Fresh Creek Nature Preserve': 693, 'Puget Creek Natural Area': 694, 'East Madison Street Ferry Dock': 695, 'Bhy Kracke Park': 696, 'Beacon Hill Playground': 697, 'Olympic Hills Es Playfield': 698,
                  'Orchard Street Ravine': 699, 'New Star Fish Market': 700, 'RAIN Inwood Neighborhood Senior Center': 701, 'Halal Street Meat Cart': 702, 'Lt Joseph Petrosino Park': 703, 'Thorndyke Park': 704,
                  'Pratt Park': 705, 'Broadway': 706, 'Cleveland Playfield': 707, "Terry's Thermador Appliance Experts": 708, 'Seattle Appliance Specialist': 709, "Alex's Watch & Clock Repair": 710,
                  'Washington Care Center': 711, 'New Mark Tailor': 712, 'Schatzie the Butcher': 713, 'Eatside': 714, 'Historic Bostonian Properties': 715, 'Dorchester Park': 716, 'The Courtyards at Mountain View': 717,
                  'Community College Of Aurora': 718, 'Westerly Creek Park': 719, 'New York City Hall': 720, 'Ballard Sails': 721, 'Western Avenue Senior Housing': 722,
                  'Center For Autism Rehabilitation & Evaluation': 723, 'Casa Bella Home Care Services': 724, 'Thermador Appliance Mavens': 725, 'Lake Union Yacht Center': 726, 'Emerald Harbor Marine': 727,
                  'Sandel Playground': 728, 'Rubber Chicken Museum': 729, 'Ashburton Place Plaza': 730, 'Copley Square Park': 731, 'Walden Street Community Garden': 732, 'Railroad Avenue': 733,
                  'Minton Stable Garden': 734, 'Washington Park Arboretum': 735, 'Seattle Municipal Tower': 736, 'Jefferson Park Golf Course': 737, 'P S 75 Robert E Peary': 738, 'East River Waterfront Esplanade': 739,
                  'Judge Charles M Stokes Overlk': 740, 'Plymouth Pillars Park': 741, 'Ne 60Th Street Park': 742, 'Twelfth West And W Howe Park': 743, 'Equity Park': 744, "DU's historic Chamberlin Observatory": 745,
                  'Alcohol Drug Rehab Denver': 746, 'The Charles River': 747, 'Comm of Mass Dcr': 748, 'Eastern National': 749, 'New England Aquarium Whale Watch': 750, 'Kennedy Library Harborwalk': 751,
                  'Harambee Park': 752, 'Mass Art Campus': 753, 'Bible James A Park': 754, 'Sabin': 755, 'Inspiration Point Park': 756, 'City Of Karmiel Park': 757, 'Heritage Park': 758, 'Virginia Park': 759,
                  'Metro Park': 760, 'Flourish Supportive Living': 761, 'Hentzel Paul A Park': 762, 'Solstice Park': 763, 'Odyssey Sober Living': 764, 'East Village Meat Market': 765, 'Denison Park': 766,
                  "Gilda's Club New York City": 767, "Raul's Viking Repair Services": 768, 'Sky View Observatory': 769, 'Highland Park': 770, 'Chappetto Square': 771, 'View Ridge Playfield': 772,
                  'Kerry Park Franklin Place': 773, 'Appliance Masters In Seattle': 774, 'Neponset Valley Parkway': 775, 'Kidsport': 776, 'European Senior Care': 777, 'Allen Mall One': 778, 'Ella Bailey Park': 779,
                  'Space Needle': 780, 'Soundview Playfield': 781, 'Indie Fresh': 782, 'Pritchett Island Beach': 783, 'Somerset Street Plaza': 784, 'Oak Square': 785, 'Coppens Square': 786, 'Starfire Education': 787,
                  'Hot Wheels Auto Body': 788, 'Envirotech Toyota Lexus Scion Independent Auto': 789, 'Hair By Andy': 790, 'Bronstein & Associates Insurance Brokers': 791, 'The Goodlife Nutrition Center': 792,
                  'Speedy Auto & Window Glass': 793, 'Metro United Bank': 794, 'Tamarind Financial Planning': 795, 'Salon Kavi': 796, 'Townhouses at 3111 S Delaware St': 797, 'HarborView Park': 798,
                  'Garden of Peace': 799, 'Hennigan Schoolyard': 800, 'Umass Boston': 801, 'Blue Hill Rock': 802, 'Applied Strategies': 803, 'The Peninsula Garage Door': 804, 'Makeover Galore': 805, 'SFOcloud': 806,
                  'Lear': 807, "Ma's Auto Body": 808, 'Clothesline Laundromat': 809, 'Edgar Allan Poe Statue': 810, 'Jamaica Pond Park': 811, 'Fort Warren': 812, 'Kenmore Square': 813,
                  'Massachusetts College of Art and Design': 814, 'Nell Singer Lilac Walk': 815, 'Sea Breeze Fish Market': 816, 'Queen Ann Drive Bridge': 817, 'Baker Park On Crown Hill': 818, 'Laguna Vista': 819,
                  'Allpoint ATM': 820, 'Win Door Service': 821, 'Sturge Presbyterian Church': 822, 'My Pro Auto Glass': 823, 'Total Wine & More': 824, 'Institute on Aging San Mateo County': 825,
                  'Living Word Chruch Of The Deaf Incorporated': 826, 'Aikido By the Bay': 827, 'National Museum Of Hip Hop': 828, "Wang's Chinese Medicine Center": 829, 'James Manley Park': 830,
                  'Heritage Club At Denver Tech Center': 831, "St Lucy's Schl": 832, 'Alpha Beacon Christian School': 833, 'The Betty Mills Company': 834, 'Olsen Auto Body Repair': 835, "Trag's Market": 836,
                  'The Preston at Hillsdale': 837, 'Camp Couture': 838, 'Condos at 1936 Vista Cay': 839, 'Bayhill Heat & Air Inc': 840, 'Crack Corn': 841, 'Als Music Video & Games': 842, 'St Marks Greenbelt': 843,
                  'Tashkent Park': 844, 'Reel Grrls': 845, 'Quadrivium': 846, 'Creekfront Park': 847, 'Chavez Cesar E Park': 848, 'Ford Barney L Park': 849, 'Boston Long Wharf': 850, 'Truce Garden': 851,
                  'Apartments at 50 W 3rd Ave': 852, 'Apartments at 16 Hobart Ave': 853, 'Condos at 6 Seville Way': 854, "Bonardi's Vacuum & Janitorial": 855, 'Hobart Park': 856, 'Topline Automobile': 857,
                  'Bottom Line Billing Service': 858, 'Cinépolis': 859, 'A B Ernst Park': 860, 'Burke Museum Of Natural History And Culture': 861, "Edward's Refrigerator Repair Service": 862, 'Creative Date': 863,
                  'Pinehurst Pocket Park': 864, 'Bradley': 865, 'Wow Interactions': 866, 'Key Markets Main Office': 867, 'La Leena Nails': 868, "Fred's Market": 869, "Mariner's Greens Home Owners Association": 870,
                  'Shall We Dance Tango': 871, 'Apartments at 1006 Tilton Ave': 872, 'Ann Watters PhD': 873, 'Atria Senior Living': 874, 'Apartments at 200 7th Ave': 875, 'Western Graduate School Of Psychology': 876,
                  'Amos Orchids': 877, 'Marble Hill Playground': 878, 'Pralls Island': 879, 'Laviscount Park': 880, 'Warren Gardens Community Garden': 881, 'Apartments at 35 N El Camino Real': 882,
                  'Nurse Family Partnership': 883, 'Roberts': 884, 'Martin Luther King Jr Park': 885, 'Russell Square Park': 886, 'Cheasty Greenspace': 887, 'San Mateo Electronics Inc': 888, 
                  'Southwest Boston Garden Club': 889, 'Edward Everett Hale Statue': 890, 'Forsti Protection Group': 891, 'Babson Cookson Tract': 892, 'Bayview Playground': 893, 'Pot Business Rules For Washington': 894,
                  'Interbay P Patch': 895, "Diane's Alterations and Tailoring": 896, 'Lincoln Square': 897, 'Park Heenan': 898, 'Fort Independence Park': 899, 'Westlake Square': 900, 'One Three Four Elm Apartments': 901,
                  'Paris Fashion Institute': 902, "Eve's Holisitic Animal Studio": 903, 'Dachauer Sheryl M Lep Lmft': 904, 'CoinFlip Bitcoin ATM': 905, 'The Sports Museum': 906, 'Imported Foods': 907,
                  'Visage Esthetiques Skin Care': 908, 'Jacobs Frances Weisbart Park': 909, 'Lincoln': 910, 'Death Museum Store': 911, 'Volunteer Park Conservatory': 912, 'Raoul Wallenberg Forest': 913,
                  'Bay State Road': 914, 'Gas Works Park': 915, 'Ringgold Park': 916, 'Seaport Elite Yacht Charter': 917, 'Thomas C Wales Park': 918, 'East Boston Greenway': 919, 'Atlantic Avenue Plantings': 920,
                  'Tanishas Hood Drag Race': 921, 'Pike Place Market Gum Wall': 922, 'Sugar Zoo': 923, 'Hull Lifesaving Museum': 924, 'US Customs House': 925, 'Campus Convenience': 926, 'Kavod Senior Life': 927,
                  'Emeritus at Roslyn': 928, 'Downtown Childrens Playground': 929, 'Leschi Lake Dell Natural Area': 930, 'Twelfth Avenue South Viewpoint': 931, "Tom's Dog Run": 932, 'Francis Lewis Park': 933,
                  'Parkslope Eatery': 934, 'City Park': 935, 'Steck': 936, 'Shore Road Park': 937, 'Lenox Fish Market': 938, 'Hyosung Park': 939, 'Bryant Neighborhood Playground': 940, 'Court Furniture Rental': 941,
                  'Christopher Columbus Park': 942, 'St CatherineS Park': 943, 'Paerdegat Park': 944, 'Pathfinder': 945, 'Eagleton': 946, 'Montbello Civic Center Park': 947, 'B F Day Playground': 948,
                  'Touch of Elegance Events': 949, 'Fabulous Salon': 950, 'Tai Tung Brake & Muffler Auto Shop': 951, 'Mlc Financial Consulting Services': 952, 'Bay Tree Park': 953, 'Junior Gym': 954,
                  'Bermuda Apartments': 955, 'Crisis Services': 956, 'Metric Tech': 957, 'Alibaba Group': 958, 'Balance Point Strategic Services': 959, 'Hair Designs by Kelly Dinger': 960, 'Bodyrok San Mateo': 961, 
                  'San Mateo Police Department': 962, "Grain D'or": 963, 'Condos at 320 Peninsula Ave': 964, 'Jason Yui Supreme Lending': 965, 'Rose Gold Events': 966, 'Box Ship & More': 967, 'SnapLogic': 968,
                  'Jensen Instrument Company': 969, 'Silicon Valley United Church of Christ': 970, 'Apartments at 419 Rogell Ave': 971, 'Remlo Apartments': 972, 'Angel Nails': 973, 'Townhouses at 106 Franklin Pkwy': 974,
                  'Tat Wong Kung Fu Academy': 975, 'Viking Pavers Construction': 976, 'Belle Isle Coastal Preserve': 977, 'MacKenzie House': 978, 'Sunken Gardens Park': 979, 'Gust': 980, 'Rowe Street Woods': 981,
                  'Park Hill': 982, 'Montclair': 983, 'Cesare Olive Oil & Vinegars': 984, 'Kerry Park': 985, 'Always Perfect Yacht Interiors': 986, 'Golden Way Market': 987, 'Cutillo Park': 988, 'Adams Park': 989,
                  'Cuny Hunter College': 990, 'Hot Soup Cart': 991, 'Central Park Rec Center': 992, 'City Hall Plaza': 993, 'Rink Grounds': 994, 'The Nature Conservancy': 995,
                  'Massachusetts Division of Unemployment Assistance': 996, 'Coors Field': 997, 'Wildlife World Museum': 998, 'Halal Gyro Express': 999, 'Windfall Farms': 1000, 'Ramsay Park': 1001, 
                  'Rocky Mountain Arsenal': 1002, 'Maple School Ravine': 1003, 'Elitch Gardens': 1004, 'Ether Dome': 1005,'Warren Anatomical Museum': 1006, 'Ashland Recreation Center': 1007,
                  'West Roxbury H S Campus': 1008, 'Sargent Street Park': 1009, 'Lil Patch of Heaven': 1010, 'Congress Park': 1011, 'Serenity House Assisted Living Office': 1012, 'Major Mark Park': 1013, 
                  'Benedict Fountain Park': 1014, 'Dewey Square Parks': 1015, 'A MailBoxes & More': 1016, 'Trag\'s Woodwork & Designs': 1017, 'Barcelino Per Donna': 1018, 'Apartments at 2645 S El Camino Real': 1019, 
                  'Centerwood Liquors': 1020, 'Yan\'s Skin Care': 1021, 'Elaine T Valente Open Space': 1022, 'Assited Living Denver': 1023, 'Cherry Creek Greenway': 1024, 'Stevens Triangle': 1025,
                  'Dick Paetzke Creative Directions': 1026, 'Amusement Services': 1027, 'Kilbourne Park': 1028, 'Pioneer Association of the State of Washington': 1029, 'Smith Cove Park': 1030, 'Mary Soo Hoo Park': 1031,
                  'Living Computer Museum': 1032, 'Washington Park Japanese Gdn': 1033, 'Village Place Park': 1034, 'Sonny Lawson Park': 1035, 'University Of Colorado Museum Of Natural History': 1036, 
                  'The Yearling': 1037, 'W H Ferguson Park': 1038, 'Gates Crescent Park': 1039, 'Lucy\'s Whey': 1040, 'Cheltenham': 1041, 'Denver Housing Authority': 1042, 'Weir Gulch Marina Park': 1043, 
                  'Huston Lake Park': 1044, 'Denver Kickball Coalition': 1045, 'Museum Of Anthropology': 1046, 'Carlton Park Foster': 1047, 'Halal Food Cart': 1048, 'Sign Language Center': 1049, 'Mca': 1050, 
                  'EcoGuard Appliance': 1051, 'Waterfront Park': 1052, 'Doors Open Denver': 1053, '82nd Street St Stephen\'s Greenmarket': 1054, 'Ruby Hill Park': 1055, 'Senior Housing Options The Barth Hotel': 1056, 
                  'Washington Park Boathouse': 1057, 'Pelham College': 1058, 'The Boat Rental': 1059, 'Podium Plaza': 1060, 'Counterbalance Park': 1061, 'Spectrum Seniors': 1062, 'Trans World Life Insurance Company': 1063, 
                  'Inpowerfit': 1064, 'Brown William Hair Stylists': 1065, 'Fitness Therapy': 1066, 'Prexion': 1067, 'San Mateo County Community College District Office': 1068, 'Baywood Oaks': 1069, 
                  'Mani Cute Nail Spa': 1070, 'Jiffy Lube': 1071, 'San Mateo Foster City Schl Dist Horrall School': 1072, 'Tian Yuan Taoist Temple': 1073, 'Delta Electric Company': 1074, 'Les Amis Salon et Spa': 1075, 
                  'Quality Towing': 1076, 'Allwin Capital': 1077, 'Boostan Kids': 1078, 'Quincy Street Play Area': 1079, 'Boston University School of Theology': 1080, 'Stonehill Park': 1081, 
                  'Isabella Stewart Gardner Museum': 1082, 'The Paul Revere House': 1083, 'Commercial Point': 1084, 'Mcgann Park': 1085, 'Franklin Park': 1086, 'Msgr Francis A Ryan Park': 1087, 'Newbury Street': 1088, 
                  'Tree Sculpture': 1089, 'Willowwood Rock': 1090, 'Sullivan Square': 1091, 'Geneva Cliffs': 1092, 'Marsh Plaza': 1093, 'University Heights': 1094, 'Old City Hall': 1095, 'Brighton Common': 1096, 
                  'Thea and James M Stoneman Centennial Park': 1097, 'William Ellery Channing Monument': 1098, 'Mount Baker Boulevard': 1099, 'Centennial Park': 1100, 'Good Will Hunting Bench': 1101, 
                  'Permanent Mission of Cuba To the United Nations': 1102, 'Kitchen Masters Specialist': 1103, 'Amira Acne Solutions': 1104, 'Monkey Fist Marine': 1105, 'Rainier Beach Playfield': 1106,
                  'Hunger Intervention Program': 1107, 'Attorney General': 1108, 'Rocky Mountain Arsenal National Wildlife Refuge': 1109, 'Pipers Creek Natural Area': 1110, 'Throgs Neck Park': 1111, 
                  'Fort Point Associates': 1112, B'nan Parcel': 1113, 'Joslin Park': 1114, 'Adventurers Family Entertainment Center': 1115, 'Constitution Plaza': 1116, 'Square Park': 1117, 'Flores Hector M Park': 1118,
                  'West Thames Dog Park': 1119, 'Atlas Autoglass': 1120, 'A 1 Overhead Door Company': 1121, 'Sofia De La Vega': 1122, 'Mother Brook Iii': 1123, 'Codman Square': 1124, 'Thrifty Janitorial': 1125, 
                  'Belltown Cottage Park': 1126, 'Western Motorsports': 1127, 'Northfield Athletic Complex': 1128, 'Mission Main Playlots': 1129, 'Boston University Club': 1130, 'Gerberding Hall': 1131, 
                  'Wood Island Bay Marsh': 1132, 'Brookdale Denver Tech Center': 1133, 'Ashford University': 1134, 'Ny City': 1135, 'Doug\'s Truck & Equipment Repair': 1136, 'Dean Harman': 1137, 
                  'Thrive Integrative Health': 1138, 'C H Robinson Worldwide': 1139, 'Evergreen Towing Service': 1140, 'St James African Methodist Episcopal Zion Church': 1141, 
                  'Apartments at 3828 3830 Colegrove St': 1142, 'Bromwell': 1143, 'Overland Pond Park': 1144, 'Garrison And Union Park': 1145, 'Aztlan Park': 1146, 'Urban Assault Ride by New Belgium': 1147, 
                  'Columbia Rehabilitation and Regenerative Medicine': 1148, 'Intresys TurboCourt': 1149, 'Apartments at 32 N Ellsworth Ave': 1150, 'Carlyle Jewelers': 1151, 'Han\'s Asian Antiques & Arts': 1152, 
                  'TLC Clean & Sober Living Homes': 1153, 'All Pro Plumbing': 1154, 'Nielle Styles': 1155, 'Amino Kit': 1156, 'Technical Information Associates': 1157, 'Lamber Thermador Range Repair Service': 1158,
                   'Metro Tailoring & Alterations': 1159, 'Ridge Custom Upholstery': 1160, 'Barr Dave Plumbing': 1161, 'Yijuan Better Health Clinic': 1162, 'Seattle Public Schools': 1163,
                   'Budget Blinds of West Seattle & Central South': 1164, 'Colorado Film School': 1165, 'Colorado Museum Of Natural History': 1166, 'Fairmont': 1167, 'Kiwanis Ravine Overlook': 1168,
                   'Brighton Police Station Campus': 1169, 'Greenwood Triangle': 1170, 'Sutherland Cooktop Repair': 1171, 'S Lyons': 1172, 'Sorrentino\'s Market': 1173, 'Laredo Avenue Parcel': 1174, 
                   'Juniper Valley Park': 1175, 'Pivot Art Culture': 1176, 'Solo Appliance Repair': 1177, 'Highland Gateway Park': 1178, 'Better Today Addiction Rehab': 1179, 
                   'Independent Growth Home Healthcare Agency': 1180, 'Flushing Greens': 1181, 'Cherry Creek Recreation Area': 1182, 'United States Mint': 1183, 'Mitchell': 1184, 'Bellevue Hill Reservation': 1185,
                   'Cook Park Recreation Center': 1186, 'Independence House Northside': 1187, 'Independence Plaza Parking': 1188, 'East 86Th Ave Trail Greenway': 1189, 'SloanS Lake Park': 1190, 
                   'United Drug Rehab Group Denver': 1191, 'Munroe': 1192, 'Teller': 1193, 'Dickson\'s Fine Brines': 1194, 'Queens Botanical Garden Farmers Market': 1195, 'Pershing Square Plaza': 1196, 
                   'Alexander Hamilton Statue': 1197, 'Traylor': 1198, 'Mestizo Curtis Park': 1199, 'Coast Guard Museum': 1200, 'Slope Park': 1201, 'Sole Repair': 1202, 'Highland Drive Parkway': 1203, 
                   'Roanoke Park Watch Repair': 1204, 'Minetta Triangle': 1205, 'Sandstone Care': 1206, 'Riverfront Park': 1207, 'Flushing Airport': 1208, 'Forest Hills Station Mall': 1209, 'Franklin Park Zoo': 1210, 
                   'Harvard Stadium': 1211, 'West End Museum': 1212, 'Whittier': 1213, 'Marrama': 1214, 'Newlon': 1215, 'China Tea Herbs & Health Products': 1216, 'Spangler Food Court at Harvard Business School': 1217, 
                   'Ronan Park': 1218, 'Boston University Academy': 1219, 'Old North Church Museum': 1220, 'Frieda Garcia Park': 1221, 'Adolescent Psychiatry & Pediatric Psychiatry': 1222, 'Barrett': 1223, 
                   'St PatrickS Neighborhood Park': 1224, 'Little Mystic Access Area': 1225, 'Observatory Park': 1226, 'John F Kennedy Presidential Library And Museum': 1227, 'United States Government': 1228, 
                   'Winthrop Square': 1229, 'Huntington Hemenway Mall': 1230, 'Martini Shell Park Neponset River Reservation': 1231, 'Stanley Ringer Playground': 1232, 'Fenway Park': 1233, 'Stuyvesant Organic': 1234, 
                   'De Witt Clinton Park': 1235, 'Equity Point': 1236, 'Strandway Castle Island': 1237, 'Citgo Sign': 1238, 'Bostonian Society Museum Shop': 1239, 'Lovells Island': 1240, 'Copley Place Plaza': 1241, 
                   'National Consumer Law Center Inc': 1242, 'SoldierS Monument': 1243, 'Flaherty Park': 1244, 'James P Kelleher Rose Garden': 1245, 'African Burial Ground National Monument': 1246, 'Holm': 1247, 
                   'South Bark Dog Park': 1248, 'GameWorks': 1249, 'Tortoise and the Hare Sculpture': 1250, 'Woodcliff': 1251, 'Hallett Fund Academy': 1252, 'A Caring Hand': 1253, 'Ue Enterprises': 1254, 
                   'WolfeS Pond Park': 1255, 'Mott St Senior Center': 1256, 'Aviation & Space Center of the Rockies': 1257, 'Westerly Creek': 1258, 'St Mark\'s Church Greenmarket': 1259, 'illume Electrolysis': 1260, 
                   'Apartments at 739 Highland Ave': 1261, 'Lot Surf': 1262, 'The Big Dig': 1263, 'Boston Nature Center': 1264, 'Belle Isle Marsh Reservation': 1265, 'Fisher Pavillion': 1266, 'Mattahunt Woods Iii': 1267,
                   'Brewer Burroughs Tot Lot': 1268, 'Chinese Historical Society of New England': 1269, 'Williams Fruit Farm': 1270, 'World Trade Center Memorial Foundation': 1271, 'Pferdesteller Park': 1272, 
                   'MGA Crisis Intervention': 1273, 'Denver Phlebotomy School': 1274, 'Berkeley Leashless Dog Park': 1275, 'Sullivan Gateway': 1276, 'Drew University U N Program': 1277, 'Pearly Jones Home': 1278, 
                   'Interbay Golf': 1279, 'Coffee Physics': 1280, 'VR Arcade': 1281, 'Johnson': 1282, 'Burns Dc Park': 1283, 'Cook Judge Joseph E Park': 1284, 'Viking Park': 1285, 'Marston Lake Walking Trail': 1286, 
                   'University Of Denver Pioneers Men\'s Basketball': 1287, 'Decatur West Personal Care': 1288, '5th Avenue': 1289, 'High Line Canal Greenway': 1290, 'Mimi': 1291, 'Sail Park': 1292, 
                   'National American University': 1293, 'Colorado Parks & Wildlife': 1294, 'Apartments at 3120 Los Prados St': 1295, 'Maven Lane Financial Group': 1296, 'Massachusetts Korean War Veterans Memorial': 1297,
                   'Life AcuMed Spa': 1298, 'Bal Aire Mechanical': 1299, 'Eastmoor Park': 1300, 'Harriet Tubman Memorial Statue': 1301, 'Heart 2 Heart 4 Seniors': 1302, 'Scottish Angus Cow and Calf': 1303, 
                   'Stand in Opposition Plaque': 1304, 'Jewels of Highlands House Tour': 1305, 'Carl Schurz Big Dog Run': 1306, 'Hutchinson Theodore A Park': 1307, 'Rosemark At Mayfair Park': 1308, 
                   'City Of Axum Park': 1309, 'Bear Creek Park': 1310, 'Leland Street Herb Garden': 1311, 'School Of The Museum Of Fine Arts At Tufts University': 1312, 'The Humps': 1313, 'Dragonfly Pavilion': 1314, 
                   'Waterway': 1315, 'Harvard Gulch Golf Course': 1316, 'George Washington Bridge Park': 1317, 'Denver School Of Nursing': 1318, 'Marycrest Assisted Living': 1319, 'Mile High Youth Corps': 1320, 
                   'X Gaming': 1321, 'Commons Park': 1322, 'Castro': 1323, 'Molly Brown House Museum': 1324, 'D C Burns Park': 1325, 'Outpatient Women\'s Treatment Services': 1326, 'Yeshiva Toras Chaim': 1327, 
                   'Kirsten Kell Marriage and Family Therapist': 1328, 'Home Instead Senior Care': 1329, 'Deutscher Advisory Group': 1330, 'Apartments at 216 226 6th Ave': 1331, 'Nitsa\'s Barbering': 1332, 
                   'Tech Rocks': 1333, 'Kelly Services': 1334, 'The Children\'s Museum of Denver at Marsico Campus': 1335, 'Cuatro Vientos Four Winds Park': 1336, 'Amesse': 1337, 'Sweet Town': 1338, 
                   'Marion Street Community Center': 1339, 'Shiki Dreams': 1340, 'French & Associates': 1341, 'John Jay Park': 1342, 'Office Depot': 1343, 'The Support Group': 1344, 
                   'Apartments at 120 Peninsula Ave': 1345, 'Great Kills Park Hillcrest': 1346, 'Shivkrupa': 1347, 'Red Auerbach Concourse': 1348, 'Seymour Deborah J Psyd': 1349, 'Archuleta': 1350, 
                   'Bunker Hill Cc Campus Grounds': 1351, 'Shirley Eustis House': 1352, 'Lavietes Pavilion': 1353, 'Columbia Park': 1354, 'Rev Loesch Family Park': 1355, 'Gibson House Museum': 1356, 
                   'Bill Russell Statue': 1357, 'Otis House': 1358, 'Forbes Street Garden': 1359, 'University Of Massachusetts Boston': 1360, 'Stapleton Development Corporation': 1361, 'Pier 10 Mall': 1362, 
                   'Boylston Place': 1363, 'Nordic Museum': 1364, 'Frink Park': 1365, 'City Of Potenza Park': 1366, 'Manley James N Park': 1367, 'Washington Park JJ Byrne Playground': 1368, 
                   'Lavilla Meadows Na On Tc': 1369, 'Fairview Walkway': 1370, 'Miracle Thai': 1371, 'The Dunes Rehab NYC': 1372, 'Hybrid Oak Woods Park': 1373,
                   'Museum At Eldridge Street': 1374, 'Mcphs University': 1375, 'BU Beach': 1376, 'Agganis Arena': 1377, 'Association of Art Museum Directors': 1378, 'Union Bay Natural Area': 1379, 'Steve Herbert': 1380,
                   'Bhangra Empire Classes San Mateo': 1381, 'Js Auto Repair Center': 1382, 'Strategic Awareness Institute': 1383, 'California Catering San Mateo': 1384, 'Hauer Cathy Mft': 1385,
                   'Denise Bloom Interior Design': 1386, 'Elevate Chiropractic Studio': 1387, 'Matt Keck': 1388, 'CenturyLink Tower': 1389, 'University of Colorado At Denver and Health Sciences Center': 1390,
                   'Liberty Day': 1391, 'Thomas ONeill Tunnel': 1392, 'DC Stop Smoking': 1393, 'Mile High Continuing Care': 1394, 'Community Plaza Park': 1395, 'Harvard Gulch West': 1396, 'Lake Of Lakes Park': 1397,
                   'Montage Heights': 1398, 'The Money Museum': 1399, 'East Side Meat Corporation': 1400, 'James Madison Plaza': 1401, 'NYC Department of Parks': 1402, 'Tim\'s DCS Repair Service': 1403,
                   'Denver Roller Derby': 1404, 'Village Shoe Repair': 1405, 'Discovery Park': 1406, 'The Prince From Minneapolis Exhibit': 1407, 'Arrowhead Gardens': 1408, 'Good Shepherd Center': 1409,
                   'Molbak\'s Butterfly Garden': 1410, 'Klondike Gold Rush National Historical Park Seattle Unit': 1411, 'South Boston Maritime Park': 1412, 'Gowanus Waterfront Park': 1413, 'Barnum': 1414,
                   'City Of Chennai Park': 1415, 'Dr Jose Rizal Park': 1416, 'Essential History Expeditions': 1417, 'Westside Church Of Christ': 1418, 'Connect Wireless': 1419, 'Bay Media': 1420,
                   'Alamo Placita Park': 1421, 'Laboratory Institute of Merchandising': 1422, 'Montbello Open Space': 1423, 'Firefighters Memorial Park At Ritz Plaza': 1424, 'Our Heros Food Truck': 1425,
                   'Planeview Park': 1426, 'American Friends of the Bible': 1427, 'Boston Harbor Walk': 1428, 'Castle Square Parks': 1429, 'Prison Ship Martyrs\' Monument': 1430, 'Southwest Corridor Community Farm': 1431,
                   'Arroyo Heights': 1432, 'The Manor on Marston Lake': 1433, 'American College of Greece': 1434, 'John Harvard Mall': 1435, 'Massachusetts State House': 1436, 'Franklin Hill Green': 1437,
                   'Elliot Norton Park': 1438, 'Prince Street Park': 1439, 'USS Growler Submarine': 1440, 'Titus Sparrow Park': 1441, 'Northpoint the Evergreen Northgate': 1442, 'Classic Yacht Association': 1443,
                   'South Ship Canal Trail': 1444, 'Virgil Flaim Park': 1445, 'Seattle Drug Rehab': 1446, 'Ramler Park': 1447, 'Childe Hassam Park': 1448, 'University of Waterloo Manhattan': 1449,
                   'Thrive Center For Healing Traditions': 1450, 'Fairview Park': 1451, 'Tepper Triangle': 1452, 'National Race Masters': 1453, 'William F Passannante Ballfield': 1454, 'Village Pediatrics': 1455,
                   'Revere Plaza': 1456, 'Village Care of New York': 1457, 'Force Basketball': 1458, 'Mcglone': 1459, 'Police Athletic League': 1460, 'City View Park': 1461, 'Wisteria Park': 1462,
                   'Linden Orchard Park': 1463, 'Hornblower Yachts': 1464, 'Asia Society and Museum': 1465, 'African Burial Ground Memorial Office': 1466, 'Owen\'s Candy Stand': 1467, 'Colorado Democratic Party': 1468,
                   'Central Park': 1469, 'Green Valley Ranch East Park': 1470, 'La Salle Deli': 1471, 'Safe Harbor Home Care': 1472, 'Jamaica Plain': 1473, 'Antioch University Seattle': 1474,
                   'Sugar Foods Corporation': 1475, 'Puget Sound': 1476, 'Sobel Court Park': 1477, 'P S 29 Ballfield': 1478, 'Institute of Electrical & Electronic Eng': 1479, 'American Musical And Dramatic Academy': 1480,
                   'Champion Small Engine Repair': 1481, 'Pretzel Time': 1482, 'Soliant Consulting': 1483, 'Lilani Wealth Management': 1484, 'Mind Your Stories & Live Well': 1485, 'Panaderia Guatemalteca Tikal': 1486,
                   'Transportes Ramos': 1487, 'Apartments at 1415 Palm Ave': 1488, 'Hallmark Cards': 1489, 'WinnowPro': 1490, 'Just For Optical': 1491, 'Banana Republic': 1492, 'Bay Meadows East': 1493,
                   'James H Platt Park': 1494, 'Museum Of Computer Technology': 1495, 'Boxer Specialist': 1496, 'Symbolscape Media': 1497, 'Cu Art Museum': 1498, 'San Mateo Auto Care': 1499,
                   'Sharleen\'s Tutoring Service': 1500, 'Macintosh Kenneth M Park': 1501, 'Ravenna Woods': 1502, 'Leventhorp Behavioral': 1503, 'Quincy Coleman Garden': 1504, 'Wartburg College West': 1505,
                   'Overland Park Golf Course': 1506, 'Katie Blacks Garden': 1507, 'Battery Park City Tennis': 1508, 'South Passage Point Park': 1509, 'Alki Point Lighthouse Tours': 1510, 'American Firefighter': 1511,
                   'Transition & Recovery Counseling Center': 1512, 'Rose Garden at Woodland Park Zoo': 1513, 'Macaron Day': 1514, 'Museum Of Holography': 1515, 'The Pearly Gates': 1516, 'Quality Hill Park': 1517,
                   'Parkfield': 1518, 'Boren Park': 1519, 'West Queen Anne Playfield': 1520, 'Northgate Plaza': 1521, 'Terminal 18 Park': 1522, 'KaboomVR': 1523, 'City Square': 1524, 'Washington Huskies Football': 1525,
                   'Balfour Riverfront Park': 1526, 'Churro & Elote Esquite Cart': 1527, 'Western Safety Products Inc': 1528, 'Stonepath Logistics Int\'l Svcs': 1529, 'Divine Hospice & Palliative Care': 1530,
                   'Westwood Park': 1531, 'NYBG Greenmarket': 1532, 'Cottage Hill Senior Apts': 1533, 'The Gardens At St Elizabeth': 1534, 'Palmer': 1535, 'Boat Electric': 1536, 'Martha Washington Park': 1537,
                   'Denver Men\'s Hockey': 1538, 'The Bicycle Repair Shop': 1539, 'Barnum Dog Park': 1540, 'Denver City And County Government Museums': 1541, 'Grant Frontier Park': 1542, 'A Leg Up': 1543,
                   'Martin Luther King Park': 1544, 'Boyd J Langston Park': 1545, 'Children\'s Hospital Colorado KidStreet': 1546, 'Skyline Park': 1547, 'Epoch Assisted Living of Denver': 1548,
                   'Valverde Park': 1549, 'WorldBest Seattle Viking Range Repair': 1550, 'La Familia Recreation Center': 1551, 'Coral Sphere': 1552, 'Espresso Technolgies': 1553, 'Red Square': 1554,
                   'Sandos M L Sam Park': 1555, 'Elenvilla Alterations & Tailoring': 1556, 'Steamworks Espresso Service': 1557, 'Wallingford Steps': 1558, 'Kitchen Repair Specialists': 1559,
                   'Buck Creek Presbyterian Camp': 1560, 'Crown Hill Park': 1561, 'Uysal Alterations': 1562, 'Fremont Canal Park': 1563, 'Central Park Stapleton': 1564, 'Welch Park': 1565,
                   'Lilu Star Indoor Playground': 1566, 'Harvard Gulch Park': 1567, 'E & V Autosales': 1568, 'Sanderson Gulch': 1569, 'Edward Everett Square': 1570, 'Navy Recruiting District Seattle': 1571,
                   'Naval Reserve Armory': 1572, 'JLA Meat Market': 1573, 'Pat Hoffman Friedman Playground Central Park': 1574, 'Ronnybrook Farm Dairy': 1575, 'Motif Investing': 1576, 'Mortgage World': 1577,
                   'Ice Oasis San Mateo': 1578, 'Second Baptist Church of San Mateo': 1579, 'Hari Om Shiri Raadha Krishna Mandir': 1580, 'Sutter Care At Home San Mateo': 1581, 'Franklin Templeton San Mateo': 1582,
                   'Carey School The': 1583, 'Sprint Cellular Service': 1584, 'Honeybee Tattoo': 1585, 'Care Resource Network': 1586, 'Gemenskap Park': 1587, 'The Watch & Jewelry Repair': 1588, 'The ROM Clinic': 1589,
                   'Nob Hill Pizza': 1590, 'CoppS Hill Terrace': 1591, 'Cedar Square': 1592, 'USS Cassin Young': 1593, 'Seaport': 1594, 'iCute iLashes': 1595, 'VR Visage': 1596, 'Lee\'s Comics': 1597,
                   'Candle House': 1598, 'Gateway Park': 1599, 'Primal Pet Foods': 1600, 'Storii Inc': 1601, 'Progressive International Motorcycle Shows': 1602, 'Lead Title Loans': 1603, 'Tom\'s Towing': 1604,
                   'Zutopia Store Hillsdale Mall': 1605, 'State Farm': 1606, 'Andrew Krzeptowski': 1607, 'Metropolitan Apartments': 1608, 'La Hacienda Market': 1609, 'Sears Portrait Studio': 1610,
                   'Shakespeare Garden': 1611, 'Volunteer Parkway': 1612, 'Casa De Dios Puerta Del Cielo': 1613, 'Copps Hill Burying Ground': 1614, 'Shane\'s Barbershop & Shave Parlour': 1615,
                   'Thermador Repair Group Seattle': 1616, 'Housing & Urban Development': 1617, 'Bastion Construction Services': 1618, 'James Hayes Park': 1619, 'Project Mindfull': 1620,
                   'High Amplitude Health Chiropractic and Sports Injury': 1621, 'East Village Cheese': 1622, 'Vinmont Veteran Park': 1623, 'Veterans Memorial Park': 1624, 'Centennial Gardens': 1625,
                   'Fresh&co.': 1626, 'Emeritus at Highline': 1627, 'JP Custom Upolstery and Repair': 1628, 'Queen Anne': 1629, 'Six Diamonds Park': 1630, 'Eckerton Hill Farm': 1631, 'Cranmer Park': 1632,
                   'Verdi Square': 1633, 'Matthews Beach Park': 1634, 'Hutchinson Park': 1635, 'Southwest Corridor Park': 1636, 'Urbani Truffles': 1637, 'Seattle Selfie Museum': 1638, 'Mr. Appliance': 1639,
                   'Seattle Metro Appliance Repair': 1640, 'Luna Park Hoa': 1641, 'Gimbels Traverse': 1642, 'Poseidon Bakery': 1643, 'H & Y Grocery Store': 1644, 'University House Wallingford': 1645,
                   'Project Renewal Clinton Residence': 1646, 'Stanley Sayres Memorial Park': 1647, 'Licton Springs Park': 1648, 'Little Brook Park': 1649, 'Ciancio Park': 1650, 'North Beach Es Playfield': 1651,
                   'Museum Of Communications': 1652, 'Centflor Mfg': 1653, 'Seattle The Top Repair': 1654, 'Max Delivery': 1655, 'Kissena Park': 1656, 'Georgetown Powerplant Museum': 1657,
                   'The Boston Common Frog Pond': 1658, 'Five Points Historic Cultural District': 1659, 'College Point Park': 1660, 'Boston Business School': 1661, 'Herter Park': 1662, 'Athena University': 1663,
                   'Alice Austen Park': 1664, 'Lake City Watch Repair': 1665, 'Ticketmaster': 1666, 'Constitution Beach': 1667, 'Hamilton Viewpoint Park': 1668, 'Einstein Mural by Kobra': 1669,
                   'Lakewood Playfield': 1670, 'Boston Fire Museum': 1671, 'Lincoln Park': 1672, 'Mt Pleasant Play Area': 1673, 'College View': 1674, 'Colorado Heights University': 1675,
                   'Massachusetts Historical Society': 1676, 'Nichols House Museum': 1677, 'Boston Marathon Attack memorial': 1678, 'Skywalk Observatory': 1679, 'Denver Technological Center': 1680,
                   'Filipino American National Historical Society': 1681, 'Curtis Park': 1682, 'Cal Anderson Park and Bobby Morris Playfield': 1683, 'Ballard Swimming Pool': 1684, 'Granada Park Garden': 1685,
                   'Meat Corporation Luis': 1686, 'Best Buy Shellfish': 1687, 'Colfax': 1688, 'Discover the Dinosaurs': 1689, 'Magnolia Park': 1690, 'Dearborn Park': 1691, 'Delridge Playfield': 1692,
                   'Boyer Thermador Appliance Repair': 1693, 'Maple Leaf Reservoir Park': 1694, 'Lakeridge Playfield': 1695, 'Chelsea Street Bridge': 1696, 'USS Constitution': 1697,
                   'Pacific Northwest Museum Of Motorcycling': 1698, 'Marine Corps Recruiting': 1699, 'Madrona Playground': 1700, 'Euston Path Rock': 1701, 'Sanderson Gulch Park & Trail': 1702,
                   'Maple Wood Playfield': 1703, 'Busy Shoes Quality Shoe Repair': 1704, 'Lakewood Moorage': 1705, 'Sweet Gifts Video Cafe': 1706, 'High Line Art': 1707, 'Distortions Monster World': 1708,
                   'Lake City Young People': 1709, 'Logan Field': 1710, 'Seacrest Park': 1711, 'Williamsbridge Oval Park Dog Run': 1712, 'Ward Springs Park': 1713, 'Andover Place': 1714, 'Medal Of Honor Park': 1715,
                   'Brooklyn Academy Of Music': 1716, 'Sub Zero Quality Prime Repair': 1717, 'Concourse Ticket Agency': 1718, 'Allstate Insurance': 1719, 'Kim\'s Nails': 1720, 'A World Driving School': 1721,
                   'San Mateo Farmers\' Market': 1722, 'Mills Peninsula Behavioral Health Services': 1723, 'Zen GE Monogram Repair': 1724, 'Thomas Street Mini Park': 1725, 'Sibby\'s Cupcakery': 1726,
                   'Suffolk University': 1727, 'Arroyos Natural Area': 1728, 'Park Joon': 1729, 'Aquatics Unlimited Instructional Products': 1730, 'The Esplanade': 1731, 'Petals Florist': 1732,
                   'Hat n\' Boots': 1733, 'Harvard Business School Press': 1734, 'Amelia': 1735, 'Permanent Mission of Portugal To the United Nations': 1736, 'United American Bank': 1737,
                   'Flexible Office Space at 999 Baker Way': 1738, 'Fred Tea Shop': 1739, 'Creative Energy': 1740, 'Philbrick Schoolyard': 1741, 'Masonic Hall': 1742, 'Museum Of Fine Arts': 1743,
                   'Walrus Machine': 1744, 'Coronado Pkwy': 1745, 'Altura Park': 1746, 'Chaffee Park': 1747, 'City Market Cafe': 1748, 'Boricua College': 1749, 'McCarren Softball Diamond': 1750,
                   'Columbian': 1751, 'Denver Firefighters Museum': 1752, 'Wallingford Playfield': 1753, 'Royce\' Chocolate': 1754, 'Kelly Outdoor Skating Rink': 1755, 'Rockaway Beach Boardwalk': 1756,
                   'Rehab Helpline': 1757, 'Confectionery': 1758, 'Kings College': 1759, 'Commonwealth Museum': 1760, 'Giant Shoe Museum': 1761, 'Leyland Street Garden': 1762, 'Eunique Associates': 1763,
                   'Helene Madison Pool': 1764, 'Albanese Rudolph Butchr': 1765, 'Slavens': 1766, 'G B A Meats': 1767, 'Emancipation Statue': 1768, 'Smith Cove': 1769, 'Uss Constitution Museum': 1770,
                   'Wallace George M Park': 1771, 'Dolphin Fish Market': 1772, 'Higginson Schoolyard': 1773, 'Sawmill Brook Brook Farm': 1774, 'Nova Southeastern University': 1775,
                   'Kingfisher Natural Area On Tc': 1776, 'CloveS Tail': 1777, 'Smith': 1778, 'Carl Yastrzemski Statue': 1779, 'ChutneyNYC Broadway Bites': 1780, 'Mount Baker Beach': 1781,
                  'Othello Playground': 1782,'Queen Anne Swimming Pool': 1783,'Polar Bear Mechanical': 1784,'Winterdale Arch': 1785,'Dyckman Street Viaduct': 1786,'All the Comfort of Home': 1787,
                  'Cheese Unlimited': 1788,'Dcpa Sculpture Park': 1789,'Athmar Park': 1790,'National Addiction Rehabs Denver': 1791,'Columbia Tower Club': 1792,'Nantes Park': 1793,'College Street Ravine': 1794,
                  'Grady KitchenAid Repair Master': 1795,'Seattle Chinese Garden': 1796,'Admiral Farragut': 1797,'Edison': 1798,'Pagoda Park': 1799,'Boston African American National Historic Site': 1800,
                  'Malibu Beach': 1801,'American Civil Liberties Union': 1802,'Merrill Gardens at First Hill': 1803,'Daniels and Fisher Tower': 1804,'Hillside Calumet': 1805,'Conservatory Green': 1806,
                  'Forest Hills Greenspace': 1807,'Lueck Open Space': 1808,'Pino\'s Prime Meat Market': 1809,'Boston College Brighton Campus': 1810,'RosemaryS Playground': 1811,'International Process Svc': 1812,
                  'Masjid ul Haqq': 1813,'T W Automotive': 1814,'Cornish & Carey': 1815,'Tufts University Boston Campus': 1816,'Neurolink Chiropractic': 1817,'GE Monogram Appliance Repair Service': 1818,
                  'Reggie Wong Memorial Park': 1819,'West Harlem Piers': 1820,'Change': 1821,'The Sacred Church Of The Assemblies Of God': 1822,'Apartments at 245 Grand Blvd': 1823,
                  'Peninsula Custom Framing Company': 1824,'Autokraft': 1825,'Essex Property Trust': 1826,'Dino Motors': 1827,'Wooden Window': 1828,'Apartments at 231 Villa Ter': 1829,
                  'Faicco\'s Italian Specialties': 1830, 'ParrFit': 1831, 'National Ballpark Museum': 1832,'Pilates Instructor Training & PMA Certification': 1833,'Nine Rubies Knitting': 1834,
                  'Kathy McConnell': 1835,'Valentine Nails': 1836,'Fluff & Puff Dog Wash': 1837,'Swansons Shoe Repair': 1838,'Thomas Street Gardens': 1839,'Boston Tea Party Ship And Museum': 1840,
                  'Don Armeni Boat Ramp': 1841,'Brighton Hs Hillside': 1842,'Rainsford Island': 1843,'Camp Meigs': 1844,'Marvins Garden': 1845,'VNA Hospice': 1846,'Paco Sanchez Park': 1847,
                  'Porter Wound Care Center': 1848,'Compu Clases': 1849, 'Kinnear Park': 1850,'North End Park': 1851,'Orient Heights Beach': 1852,'GSVlabs': 1853,'Condos at 326 N Ellsworth Ave': 1854,
                  'Race Recycling and Compacting Equipment': 1855,'Beaver Pond Natural Area On Tc': 1856, 'New Century Hospice': 1857,'Historic Rose Garden': 1858, 'Laura\'s Alteration & Tailor Express': 1859,
                  'Cheesman Park': 1860,'Msgr Reynolds Playground': 1861,'Speer Blvd Park': 1862,'Lombardi Memorial Park': 1863,'Baywood Elementary': 1864,'Bonnie Brae Park': 1865,'Badging Office': 1866,
                  'Barrett Park': 1867,'Jefferson Square Park': 1868,'Harlem Children\'s Zone': 1869,'Niver Creek Trail Greenway': 1870,'Marky\'s on Madison': 1871, 'Goldrick': 1872, 'Fuller Park': 1873,
                  'Savin Hill Beach': 1874,'Majestic Heightz': 1875,'Bloxton hotels': 1876,'Army National Guard': 1877,'Bay Area Disc Centers': 1878,'Clock Repair Team': 1879,'Apartments at 815 E 4th Ave': 1880,
                  'Henry Brooks Senior Housing': 1881,'Old North Memorial Garden': 1882,'Wabash Trailhead': 1883,'First Hill Park': 1884,'VCA Animal Hospitals': 1885,'University Lake Shore Place': 1886,
                  'Fan Pier Park': 1887,'Mattahunt School Woods': 1888,'Boston Architectural College': 1889,'Kelly Open Space': 1890,'El Ciro Fish Market': 1891,'Froula Playground': 1892,'Zuni Park': 1893,
                  'Sandhofer Reservoir': 1894,'Aspgren Clifford Park': 1895,'Cambridge College Boston': 1896,'Boston Children\'s Museum': 1897,'Denver Outdoor Adventure Company': 1898,
                  'RMD Goodyear Tire Center': 1899,'Haven Avenue': 1900,'Fred Thomas Park': 1901,'Ash Grove Park': 1902,'Freedom Debt Relief': 1903,'Simon Poon At Maison Salon': 1904,'LGBT Memorial': 1905,
                  'Washington Trust For Historic Preservation': 1906,'Carleton Court Dog Park': 1907,'Mattahunt Schoolyard': 1908,'Little Boxcar Dog Park': 1909,'Rocky Mountain Lake Park': 1910,
                  'Cherry Tree Park and Playground': 1911,'Homer Harris Park': 1912,'Mamak Cart': 1913,'Manhattan By Sail': 1914,'American Legion Playground': 1915,'Rotella Park': 1916,'Mcconnell Park': 1917,
                  'Hungarian Freedom Park': 1918,'Moscow On The Hudson': 1919,'Sterling Square': 1920,'Channel Center Dog Park': 1921,'Boston China Town Neighborhood Center': 1922,'Gansevoort Plaza': 1923,
                  'Agassiz Road': 1924,'Park Carlton': 1925,'Firehouse Mini Park': 1926,'Drug Detox Centers Denver': 1927,'Hershey\'s Chocolate World': 1928,
                  'Adventist University Of Health Sciences Denver Campus': 1929,'Kunsmiller': 1930,'Hampden Heights North': 1931,'Dent Artist Paintless Dent Removal': 1932,
                  'Proline Auto Body Service': 1933,'Kings Chapel Burying Ground': 1934,'Georges Island': 1935,'Seattle Metropolitan Police Museum': 1936,'Symphony Community Park': 1937,
                  'Museum Of American Gangster': 1938,'Denver Rehabilitation Clinic': 1939,'Swansea': 1940,'Articulated Wall by Herbert Bayer': 1941,'Pinehurst Park': 1942,'Kittredge Park': 1943,
                  'Wings Over The Rockies Air & Space Museum': 1944,'Edith Macefield\'s House': 1945,'Bow Ding Seafood': 1946,'King School Park': 1947,'Can\'t Look Away The Lure of Horror Film Exhibit': 1948,
                  'Byers Evans House Museum': 1949,'Dribbl Basketball': 1950,'HunterS Point Park': 1951,'Keycorp': 1952,'Mosaic Market': 1953,'Spaces': 1954,'Columbia University Inst Real Estate': 1955,
                  'La Alma Lincoln Park': 1956,'Dr Gene Mahaffey': 1957,'Quick Lenore B Park': 1958,'Berkeley Lake Park': 1959,'Gates Planetarium': 1960,'Walker Madame Cj Park': 1961,
                  'Seattle Children\'s Playgarden': 1962,'Me Kwa Mooks Natural Area': 1963,'Thermador Repair Group Industrial Disctrict': 1964,'Your Choice Appliance Service': 1965,
                  'Chinese Community Bulletin Board': 1966, 'Lilac Adult Family Home': 1967,'American Friends of Nishmat': 1968,'Delimanjoo': 1969,'Museum Of Mathematics': 1970,
                  'Colorado State University': 1971, 'S & A 96 Candy Store': 1972, 'Cesar Chavez Park': 1973, 'Jeep Jones Park': 1974, 'Poplar Street Play Area': 1975, 'Rutland Washington Community Garden': 1976,
                  'Creative Recreational Systems Bridgewater': 1977, 'Downtown Seattle Masters': 1978, 'Dewey Square Plaza': 1979,'Colorado Military History Museum': 1980, 'Combs Junior College': 1981,
                  'University House Resident Council': 1982,'Fremont Appliance Refit': 1983,'Harlem Shambles': 1984, 'Westlake Greenbelt': 1985,'Foster Island Wash Park': 1986, 'Zion Learning Center': 1987,
                  'Federal Reserve Plaza': 1988,'Haller Lake Community St End': 1989,'Savin Maywood Street Garden': 1990, 'Westlake Park': 1991, 'City Of Brest Park': 1992,
                  'Highlands Integrative Pediatrics': 1993,'Pike Place Urban Garden': 1994,'Mckinley Thatcher': 1995, 'Golden Triangle': 1996,'St Nicholas Park': 1997, 'Roanoke Street Mini Park': 1998,
                  'Laruleta Meat Mkt': 1999,'Rainier Valley Historical Society': 2000,'Sacred Garden Healing Center': 2001, 'Fred Lind Manor': 2002,'Broadway Hill Park': 2003, 'Capitol Hill Reservoir': 2004,
                  'Southmoor Park': 2005,'Museum Views': 2006,'Queen Anne Upholstery and Refinishing': 2007, 'Clyfford Still Museum': 2008,'9th Street Historic Park': 2009, 'Carla Madison Rec Center': 2010,
                  'Henry Art Gallery': 2011, 'Veracom Ford': 2012, 'NuevaCare': 2013, 'Community Psychiatry': 2014, 'First Republic Bank': 2015, 'Savin Hill Cove': 2016, 'Daggett Museum For Living Artists': 2017,
                  'Borough Of Manhattan Community College Of The City University Of New York': 2018, 'Odyssey': 2019, 'Elite Private Yachts New York': 2020, 'Blue Dog Pond': 2021,
                  'Montclair Recreation Center': 2022,'Empower Field at Mile High': 2023, 'Buena Vista': 2024,'Collyer Brothers Park': 2025,'Aesop Park': 2026,'Dorchester Heights Monument': 2027,
                  'The Denver Hospice': 2028, 'Jenny Manes Premier Bridal Design & Alterations': 2029, 'Campana Quetzal': 2030, 'Lewis Home Care Agency': 2031,'Meat On A Stick': 2032,'PAR3 Communications': 2033,
                  'Fenway Victory Gardens': 2034, 'Bright Landry Hockey Center': 2035,'A Night at the Museum': 2036, 'Southwest Recreation Center': 2037,'Valdez': 2038,'Centura Rehabilitation': 2039,
                  'Hubbard Homestead Park': 2040, 'Gourmet Garage': 2041,'Westside Market NYC': 2042, 'Massachusetts Avenue Malls': 2043, 'Reliable Corporation Lobby Newstand': 2044, 'Thompson Island': 2045,
                  'Lincoln Center Plaza': 2046, 'Wilis Case Golf Course': 2047,'Chef Recipes': 2048, 'Vanderbilt Park': 2049, 'Habitat Park': 2050, 'Broadway Shoe Repair': 2051, 'Elite Vend': 2052,
                  'Cherry Hill Fountain': 2053, 'Malatesta Woodworks': 2054,'Boston University School Of Medicine': 2055,'Lamps Plus': 2056,'Beacon Alterations': 2057, 'Rainbow Point': 2058,
                  'Greenwood Phinney Ruv': 2059, 'Wings Over Washington': 2060,'Alvin Larkins Park': 2061,'A & J Commissary': 2062,'Square Square': 2063, 'Belleview Suites at DTC': 2064,
                  'Huntington Learning Centers': 2065, 'Franklin Hill Bha Court': 2066,'City University Of Seattle': 2067, 'Feel Good Llc': 2068, 'Godiva Cafe': 2069, 'Boston University Hillel': 2070,
                  'Edna V Bynoe Park': 2071, 'Falcon Park': 2072, 'Paul Agustinovich PhD': 2073,'Left Coast Entertainment': 2074,'86 East Thirty Ninth': 2075,'Magnolia Care': 2076, 'Northgate Park': 2077,
                  'Emeritus At Pinehurst Park': 2078, 'Chocolat Moderne': 2079, 'Boston University Plastic & Reconstructive Surgery': 2080, 'Bearing Institute': 2081, 'Havey Beach': 2082,
                  'Horatio Harris Park': 2083, 'Colorado Access': 2084, 'Hiram M Chittenden Locks': 2085,'Confluence East Park': 2086, 'Abingdon Square': 2087, 'New Freedom Park': 2088,
                  'South Street Community Garden': 2089,'Pacific Maritime Institute': 2090, 'Hobby Lobby Stores': 2091,'John C Little Sr Park': 2092, 'Spuyten Duyvil Shorefront Park': 2093,
                  'Brophy Park': 2094, 'Meghna Grocery & Halal Meat': 2095, 'Green Lake Small Craft Center': 2096, 'Pleasant Adult Family Home': 2097,'Public Garden': 2098, 'Wheel Fun Rentals Berkeley Park': 2099,
                  'Larry Bird Plaque': 2100, 'Seattle Asian Art Museum': 2101, 'ChildrenS Park': 2102, 'T Mobile Park': 2103,'Force': 2104, 'Museum of African American History': 2105, 'NB Beauty': 2106,
                  'Williams Dr Daniel Hale Park': 2107, 'Cooper Park Dog Run': 2108, 'Integrity Motors': 2109, 'Itamae Japanese Sushi': 2110, 'College Street Park': 2111, 'Cunningham Park': 2112,
                  'Sunnyside Ave N Boat Ramp': 2113, 'Waterfalls in Central Park': 2114, 'HomeWell Senior Care': 2115,'Ballard Corners Park': 2116, 'Papabubble': 2117, 'Cherry Creek Park': 2118 }

  tl_reverse_mapping = {v: k for k, v in tl_mapping.items()}
  tl_labels = list(tl_mapping.keys())

  # Collecting user inputs
  def get_DAYOFWEEK():
    DAY_OF_WEEK = st.selectbox('Select a day of week 📆', wd_mapping)
    return DAY_OF_WEEK
  def get_TRUCK_BRAND_NAME():
      TRUCK_BRAND_NAME = st.selectbox('Select a truck brand name 🚐', bn_mapping)
      return TRUCK_BRAND_NAME
  def get_CITY():
      CITY = st.selectbox('Select a city 🏕️', ct_mapping)
      return CITY
  def get_LOCATION():
      LOCATION = st.selectbox('Select a truck location 📍', tl_mapping)
      return LOCATION  

  # Define the user input fields
  wd_input = get_DAYOFWEEK()
  bn_input = get_TRUCK_BRAND_NAME()
  ct_input = get_CITY()
  tl_input = get_LOCATION()
  
  # Map user inputs to integer encoding
  wd_int = wd_mapping[wd_input]
  bn_int = bn_mapping[bn_input]
  ct_int = ct_mapping[ct_input]
  tl_int = tl_mapping[tl_input]

  with st.sidebar:
    st.sidebar.subheader('For ease of use, stated below are some locations for each city 🗺️📍')
    st.sidebar.markdown("""
    **San Mateo** 🏞️
    - Applied Strategies (Tasty Tibs)
    - American Prime Financial (Freezing Point)
    - Remlo Apartments (Revenge of the Curds)
    
    **Seattle** ☕️
    - Waterway (Tasty Tibs)
    - Seattle Children's Playgarden (Guac n' Roll)
    - Thorndyke Park (Better Off Bread)
    
    **New York City** 🗽
    - Best Buy Shellfish (Freezing Point)
    - Gateway Newstands (Kitakata Ramen Bar)
    - Poseidon Bakery (Le Coin des Crêpes)
    
    **Boston** 🏙️
    - Bellevue Hill Reservation (Amped Up Franks)
    - Skywalk Observatory (Peking Truck)
    - Stevens Triangle (Guac n' Roll)
    
    **Denver** 🏔️
    - Pelham College (Le Coin des Crêpes)
    - Inspiration Point Park (Smoky BBQ)
    - Aviation & Space Center of the Rockies (Cheeky Greek)
    """)

  if st.button('Predict Daily Sales'):
    # Make the prediction  
    input_data = [[wd_int, bn_int, ct_int, tl_int]]
    input_df = pd.DataFrame(input_data, columns=['DAY_OF_WEEK', 'TRUCK_BRAND_NAME', 'CITY', 'LOCATION'])
    prediction = xgb_alethea.predict(input_df)   

    output_data = [wd_int, bn_int, ct_int, tl_int, prediction[0]]
    output_df = pd.DataFrame([output_data], columns=['DAY_OF_WEEK', 'TRUCK_BRAND_NAME', 'CITY', 'LOCATION', 'DAILY_SALES'])
    predicted_sales = output_df['DAILY_SALES'].iloc[0]
    st.write('The predicted daily sales is {:.2f}.'.format(predicted_sales))
  
  st.divider()
  st.title('Projected Yearly Revenue 💰')
  st.write('Sales in a city are significantly influenced by the level of urban activity, with population size being a key factor directly correlated to daily sales. :green[As city population increases, it tends to drive higher daily sales due to increased consumer demand.] Therefore, leveraging the city\'s population data, we can predict future sales trends, considering the average yearly population growth for each city in the United States of America, as reported in online sources. :blue[With this, we would be able to look at the projected sale increase for each truck brand in the cities over a span of 5 years]')
  st.write('Average yearly population growth for each city 🌇')
  st.write('San Mateo: 4,600')
  st.write('Seattle: 30,000')
  st.write('New York City: 70,000')
  st.write('Boston: 17,000')
  st.write('Denver: 34,000')

  # Load the cleaned and transformed dataset
  df2 = pd.read_csv('fs_alethea.csv')
  df3 = pd.read_csv('yr_alethea.csv')
  
  # Viewing predicted yearly sales in the future
  def get_Extra():
    YEARS = st.slider('Number of years later 📆', 1, 5, 1)
    st.write("Predicting yearly revenue in ", YEARS, 'year(s)')
    return YEARS  
  et_input = get_Extra()

  originalCity = ct_reverse_mapping[ct_int]
  originalBrand = bn_reverse_mapping[bn_int]
  
  # st.write('Current yearly revenue: {:.2f}.'.format(predicted_sales))
  if st.button('Predict Yearly Revenue'):
    futureRevenueRow = df2[(df2['CITY'] == originalCity) & (df2['TRUCK_BRAND_NAME'] == originalBrand)]
    revenueRow = df3[(df3['CITY'] == originalCity) & (df3['TRUCK_BRAND_NAME'] == originalBrand)]

    # Check that revenueRow & futureRevenueRow DataFrames are not empty
    if not revenueRow.empty and not futureRevenueRow.empty:
      # Extract information from the filtered row
      city = revenueRow['CITY'].values[0]
      truck_brand = revenueRow['TRUCK_BRAND_NAME'].values[0]
      years_revenue = revenueRow['YEARS_REVENUE'].values[0]
      city_pop = revenueRow['CITY_POPULATION'].values[0]
      # Get the column index for the future prediction
      future_column = str(et_input)
      projected_revenue = futureRevenueRow[future_column].values[0]
      # Round off the revenue values to the nearest whole number
      rounded_years_revenue = round(years_revenue)
      rounded_projected_revenue = round(projected_revenue)

      # Calculate the increase in revenue percentage
      revenue_increase_percentage = ((projected_revenue - years_revenue) / years_revenue) * 100
      rounded_increase_percentage = round(revenue_increase_percentage, 2)
      
      st.write(f'In 2022, the yearly revenue of {truck_brand} in {city} to date is ${rounded_years_revenue}, with a population of {city_pop}.')
      st.write(f"The projected yearly revenue of {truck_brand} in {city} in {et_input} year(s) would be: ${rounded_projected_revenue}, with an percentage change of {rounded_increase_percentage}% since 2022.")
       # Check if the revenue increase percentage is negative
      if rounded_increase_percentage < 0:
        st.write("Seeing that the revenue growth was not positive, this suggests that we should look into other aspects, such as the menu items offered.")
    else:
      st.write('No data found for the provided city and truck brand name.')
      

with tab2:
  # import sklearn
  # st.write(sklearn.__version__)
  st.title('Profit Prediction for Menu Items') 
  st.markdown("Complete all the parameters to predict the profit earned !")
    
  with open('profit_xinle.pkl', 'rb') as file:
      profit_xinle = pickle.load(file)
    
  with open('category_xinle.pkl', 'rb') as file:
      category_xinle = pickle.load(file)
  
  # Load the cleaned and transformed dataset
  df = pd.read_csv('profit_xinle.csv')
  
  dowMapping={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
  dowReverseMapping = {v: k for k, v in dowMapping.items()}
  dowLabels = [dowReverseMapping[i] for i in sorted(dowReverseMapping.keys())]
  
  mt_mapping = {'Ice Cream': 0, 'Grilled Cheese': 1, 'Poutine': 2,'Ethiopian': 3,'BBQ': 4,
                  'Indian': 5,'Mac & Cheese': 6,'Vegetarian': 7,'Sandwiches': 8,'Gyros': 9,
                  'Chinese': 10,'Tacos': 11,'Hot Dogs': 12,'Ramen': 13,'Crepes': 14}
  
  tbn_mapping = {'Freezing Point': 0,'The Mega Melt': 1,'Revenge of the Curds': 2,'Tasty Tibs': 3,'Smoky BBQ': 4,
                  "Nani's Kitchen": 5,'The Mac Shack': 6,'Plant Palace': 7,'Better Off Bread': 8,'Cheeky Greek': 9,
                  'Peking Truck': 10,"Guac n' Roll": 11,'Amped Up Franks': 12,'Kitakata Ramen Bar': 13,'Le Coin des Crêpes': 14}
    
  c_mapping = {'San Mateo': 0, 'New York City': 1, 'Denver': 2, 'Seattle': 3, 'Boston': 4}

  l_mapping = {'Transportes Ramos': 0, 'Yok Thai Massage': 1, "Fred's Market": 2, 'Condos at 320 Peninsula Ave': 3, 'Centerwood Liquors': 4, 'Ice Oasis San Mateo': 5, 'Viking Pavers Construction': 6, 'E & V Autosales': 7, 'Farmers Insurance Group': 8, 'Denise Bloom Interior Design': 9, 'Office Depot': 10, 'Motif Investing': 11, 'High Amplitude Health Chiropractic and Sports Injury': 12, 'CoinFlip Bitcoin ATM': 13, "Sibby's Cupcakery": 14, 'Andrew Krzeptowski': 15, 'Bridgepointe': 16, 'My Pro Auto Glass': 17, 'Jiffy Lube': 18, 'Symbolscape Media': 19, 'Autokraft': 20, 'Mlc Financial Consulting Services': 21, 'Win Door Service': 22, 'A MailBoxes & More': 23, 'RMD Goodyear Tire Center': 24, 'Associated Psychological Services': 25, 'Huntington Learning Centers': 26, 'Elevate Chiropractic Studio': 27, 'Western Graduate School Of Psychology': 28, 'Eclipz Hair Designs': 29, 'East 19th Avenue Apartments': 30, 'Golden 1 Credit Union': 31, 'Laguna Vista': 32, 'San Mateo Auto Works': 33, 'California Bank Trust': 34, 'Masjid ul Haqq': 35, 'Lily European Pedi Mani Spa': 36, 'Star Smog San Mateo': 37, 'Nine Rubies Knitting': 38, 'Gateway Park': 39, 'The Betty Mills Company': 40, 'Tina Canada Nail Technician': 41, 'Alibaba Group': 42, 'GSVlabs': 43, 'Strategic Awareness Institute': 44, 'Valentine Nails': 45, 'Just For Optical': 46, 'Dachauer Sheryl M Lep Lmft': 47, 'Yijuan Better Health Clinic': 48, 'Simon Poon At Maison Salon': 49, 'Allpoint ATM': 50, "Sharleen's Tutoring Service": 51, 'Progressive International Motorcycle Shows': 52, 'First Priority Financial': 53, 'Hobby Lobby Stores': 54, 'Essex Property Trust': 55, 'T W Automotive': 56, 'One Three Four Elm Apartments': 57, 'Creative Energy': 58, 'GreenCal Solar': 59, 'Fabulous Salon': 60, 'Apartments at 739 Highland Ave': 61, 'Integrity Motors': 62, 'Laurelwood Shopping Center': 63, 'St James African Methodist Episcopal Zion Church': 64, 'Apartments at 1415 Palm Ave': 65, 'Shore Vu Laundromat': 66, "Ma's Auto Body": 67, 'Mcc': 68, 'Townhouses at 3111 S Delaware St': 69, 'Westside Church Of Christ': 70, 'Apartments at 815 E 4th Ave': 71, 'Mortgage World': 72, 'Apartments at 16 Hobart Ave': 73, "Lee's Comics": 74, 'Bologna Chiropractic & Sports Care': 75, 'Apartments at 816 Highland Ave': 76, 'Kathy McConnell': 77, 'Atlas Autoglass': 78, 'SSB Kitchen': 79, 'Jibe Promotional Marketing': 80, '86 East Thirty Ninth': 81, 'Tracey Lactation': 82, 'Applied Strategies': 83, 'Ideal Health Clinic': 84, 'Apartments at 1116 1120 Folkstone Ave': 85, 'Thai Art of Massage': 86, 'Jolene Whitley Hair Design & Replacement': 87, 'Snap Fitness': 88, 'Key Markets Main Office': 89, 'Apartments at 602 Cypress Ave': 90, 'Bay Meadows East': 91, 'Execupark Incorporated': 92, 'Condos at 1936 Vista Cay': 93, 'Veracom Ford': 94, 'Bloxton hotels': 95, 'SFOcloud': 96, 'Apex Iron': 97, 'VIP Petcare': 98, "Shane's Barbershop & Shave Parlour": 99, 'ParrFit': 100, 'Sasi Salon': 101, 'Intresys TurboCourt': 102, 'HarborView Park': 103, 'Bermuda Apartments': 104, 'Apartments at 419 Rogell Ave': 105, 'Peninsula Golf & Country Club': 106, 'Brown William Hair Stylists': 107, 'Barastone': 108, 'Balance Point Strategic Services': 109, 'Best Western Coyote Point Inn': 110, 'NuevaCare': 111, 'Owens Electric & Solar': 112, 'Bastion Construction Services': 113, 'Hair Designs by Kelly Dinger': 114, 'Soliant Consulting': 115, 'C H Robinson Worldwide': 116, 'Angel Nails': 117, 'Visage Esthetiques Skin Care': 118, 'Macadamian': 119, 'Proline Auto Body Service': 120, 'Bal Aire Mechanical': 121, 'Apartments at 231 Villa Ter': 122, 'LifeMoves First Step for Families': 123, 'Clock Repair Team': 124, 'Jason Yui Supreme Lending': 125, 'First Republic Bank': 126, 'Kirsten Kell Marriage and Family Therapist': 127, 'Lilani Wealth Management': 128, 'Precision Concrete Cutting': 129, 'Q Fix PC Services': 130, 'Scott Eaton Supreme Lending': 131, 'Pilates Instructor Training & PMA Certification': 132, 'Bay Media': 133, 'Diamond Motors': 134, 'Lead Title Loans': 135, 'Apartments at 217 Villa Ter': 136, 'Michaels Stores': 137, 'Bhangra Empire Classes San Mateo': 138, 'Primal Pet Foods': 139, 'Amelia': 140, 'Eurasian Auto Repair': 141, 'Spectrum Seniors': 142, 'The Sacred Church Of The Assemblies Of God': 143, 'Starfire Education': 144, 'Apartments at 216 226 6th Ave': 145, 'WinnowPro': 146, 'Apartments at 731 N Amphlett Blvd': 147, 'Tai Tung Brake & Muffler Auto Shop': 148, 'Esthetique European Skin Care Clinic': 149, 'Lamps Plus': 150, 'Atria Senior Living': 151, "Trag's Woodwork & Designs": 152, 'The Peninsula Garage Door': 153, 'Sprint Cellular Service': 154, 'Petals Florist': 155, 'Js Auto Repair Center': 156, 'Total Wine & More': 157, 'Amos Orchids': 158, 'Wells Fargo ATM': 159, 'All Pro Plumbing': 160, "San Mateo Farmers' Market": 161, 'K & W Liquors': 162, 'Condos at 326 N Ellsworth Ave': 163, 'Life AcuMed Spa': 164, 'UCSF Medical Center': 165, 'Bay Area Sunrooms': 166, 'Mason James K Insurance': 167, 'Ace Hardware': 168, 'Empowerly': 169, 'San Mateo County Times': 170, 'Apartments at 1006 Tilton Ave': 171, 'Speedy Auto & Window Glass': 172, "Yan's Skin Care": 173, "Jeffery's Dog & Cat Grooming": 174, 'Apartments at 3828 3830 Colegrove St': 175, "Bo Jonsson's Foreign Car Service": 176, 'Apartments at 3120 Los Prados St': 177, 'Freedom Debt Relief': 178, 'Mark Woods BPIA Insurance': 179, 'Alpha Beacon Christian School': 180, 'Apartments at 3149 Casa De Campo': 181, 'DP Systems': 182, 'Thrive Integrative Health': 183, 'Townhouses at 106 Franklin Pkwy': 184, 'Prexion': 185, 'Delta Electric Company': 186, 'Kelly Services': 187, 'La Leena Nails': 188, 'Aikido By the Bay': 189, 'Townhouses at 821 Laurel Ave': 190, 'Reposturing The Pain Elimination Method': 191, 'Carlyle Jewelers': 192, 'Williams-Sonoma': 193, 'Connect Wireless': 194, 'The Goodlife Nutrition Center': 195, 'United American Bank': 196, 'Apartments at 321 Monte Diablo Ave': 197, 'Bronstein & Associates Insurance Brokers': 198, "Grain D'or": 199, 'Apartments at 225 Catalpa St': 200, 'Neurolink Chiropractic': 201, 'Apartments at 35 N El Camino Real': 202, 'Institute on Aging San Mateo County': 203, 'Baywood Oaks': 204, 'Apartments at 50 W 3rd Ave': 205, 'Barcelino Per Donna': 206, 'Left Coast Entertainment': 207, 'Topline Automobile': 208, 'Bayhill Heat & Air Inc': 209, 'Apartments at 603 E 5th Ave': 210, 'Tamarind Financial Planning': 211, 'Apartments at 2645 S El Camino Real': 212, 'Cornish & Carey': 213, 'San Mateo Foster City Schl Dist Horrall School': 214, 'Barr Dave Plumbing': 215, 'Muse Beverly Lmft': 216, 'San Mateo Auto Care': 217, 'Clothesline Laundromat': 218, 'Quality Towing': 219, 'Financial Title Company': 220, 'Remlo Apartments': 221, 'Bay Area Disc Centers': 222, "Tom's Towing": 223, 'Ann Watters PhD': 224, 'Y2K Nails': 225, 'Hi View Apartments': 226, 'Faenzi Associates': 227, 'San Mateo Police Department': 228, 'Hauer Cathy Mft': 229, 'Deutscher Advisory Group': 230, "Bonardi's Vacuum & Janitorial": 231, 'San Mateo County Community College District Office': 232, 'Sofia De La Vega': 233, 'Marco Nascimento Brazilian Jiu Jitsu': 234, 'Silicon Valley United Church of Christ': 235, 'Zutopia Store Hillsdale Mall': 236, 'Wooden Window': 237, 'Community Psychiatry': 238, 'Mani Cute Nail Spa': 239, 'Release From Within': 240, 'Cinépolis': 241, 'KG Fitness Studio': 242, 'RePlanet Recycling': 243, 'Home Instead Senior Care': 244, 'San Mateo Electronics Inc': 245, 'NB Beauty': 246, 'California Catering San Mateo': 247, 'Boxer Specialist': 248, 'Sutter Care At Home San Mateo': 249, 'Flexible Office Space at 999 Baker Way': 250, 'Fitness Therapy': 251, 'Standard Parking': 252, 'Hari Om Shiri Raadha Krishna Mandir': 253, "Doug's Truck & Equipment Repair": 254, 'Steve Herbert': 255, 'Gazelle Developmental School': 256, 'St Timothy School': 257, "Eve's Holisitic Animal Studio": 258, 'Tian Yuan Taoist Temple': 259, 'San Bruno Dog Obedience School': 260, 'Apartments at 20 Hobart Ave': 261, 'Swarovski': 262, 'The Highlands': 263, "Wang's Chinese Medicine Center": 264, 'Bottom Line Billing Service': 265, 'Hot Wheels Auto Body': 266, 'Evergreen Towing Service': 267, 'Trans World Life Insurance Company': 268, 'Junior Gym': 269, 'Metropolitan Apartments': 270, 'Honeybee Tattoo': 271, 'Shall We Dance Tango': 272, 'Nordstrom': 273, 'Nob Hill Pizza': 274, "Muslim Children's Garden": 275, 'King Pang Hairdressing': 276, 'Storii Inc': 277, 'Condos at 6 Seville Way': 278, 'San Mateo DMV Office': 279, 'Maven Lane Financial Group': 280, 'Come C Interiors': 281, "Alioto's Garage": 282, 'SnapLogic': 283, 'Peninsula Custom Framing Company': 284, 'Carey School The': 285, 'Box Ship & More': 286, 'Apartments at 211 E Santa Inez Ave': 287, 'Dino Motors': 288, 'Hallmark Cards': 289, 'Nielle Styles': 290, 'Mosaic Market': 291, 'Sears Portrait Studio': 292, 'Allstate Insurance': 293, 'Toi Lynn Wyle MS MFT ERYT': 294, 'Donna Ornitz': 295, "Han's Asian Antiques & Arts": 296, 'Glass Express': 297, 'Second Baptist Church of San Mateo': 298, 'U S First Federal Credit Union': 299, 'Allwin Capital': 300, "Trag's Market": 301, 'Tat Wong Kung Fu Academy': 302, 'Metric Tech': 303, 'Bay Tree Park': 304, 'American Prime Financial': 305, 'Rose Gold Events': 306, 'Franklin Templeton San Mateo': 307, 'Camp Couture': 308, 'Dent Artist Paintless Dent Removal': 309, 'A 1 Overhead Door Company': 310, 'Mills Peninsula Behavioral Health Services': 311, 'Spaces': 312, 'Project Mindfull': 313, 'Salon Kavi': 314, 'Baywood Elementary': 315, "Mariner's Greens Home Owners Association": 316, 'The ROM Clinic': 317, 'State Farm': 318, 'Matt Keck': 319, 'The Preston at Hillsdale': 320, 'Dean Harman': 321, 'Magic Mountain Playground': 322, 'Health Street': 323, 'Metro United Bank': 324, 'Reflektion': 325, 'Hair By Andy': 326, 'Casa De Dios Puerta Del Cielo': 327, 'Apartments at 200 7th Ave': 328, 'La Hacienda Market': 329, 'Apartments at 120 Peninsula Ave': 330, 'West Elm': 331, 'Sturge Presbyterian Church': 332, 'Envirotech Toyota Lexus Scion Independent Auto': 333, 'A World Driving School': 334, 'Olsen Auto Body Repair': 335, 'Bay Motors': 336, 'Inpowerfit': 337, 'Crisis Services': 338, 'Tech Rocks': 339, 'Banana Republic': 340, 'TLC Clean & Sober Living Homes': 341, 'VCA Animal Hospitals': 342, 'Mind Your Stories & Live Well': 343, 'Paul Agustinovich PhD': 344, "Kim's Nails": 345, 'Army National Guard': 346, 'Boostan Kids': 347, 'The Support Group': 348, 'illume Electrolysis': 349, 'Living Word Chruch Of The Deaf Incorporated': 350, 'Jensen Instrument Company': 351, "Nitsa's Barbering": 352, 'Les Amis Salon et Spa': 353, 'Lear': 354, 'Panaderia Guatemalteca Tikal': 355, 'Bodyrok San Mateo': 356, 'Makeover Galore': 357, 'Apartments at 32 N Ellsworth Ave': 358, 'iCute iLashes': 359, 'Aikido By The Bay': 360, 'Candle House': 361, 'Fluff & Puff Dog Wash': 362, 'Apartments at 245 Grand Blvd': 363, 'Vesey St Edible': 364, 'Lenox Fish Market': 365, 'Ny City': 366, 'Village Pediatrics': 367, 'Square Square': 368, 'Gowanus Waterfront Park': 369, 'S Lyons': 370, 'South Beach': 371, 'P S 75 Robert E Peary': 372, 'Masonic Hall': 373, 'Harlem Shambles': 374, 'Majestic Heightz': 375, 'James Madison Plaza': 376, 'I S 049 Bertha A Dreyfus': 377, 'Waldorf Schls Fund': 378, 'National Museum Of Mali': 379, 'Color Factory': 380, 'E & L Seafood': 381, 'Max Delivery': 382, 'Verdi Square': 383, 'RTV Exotics': 384, 'Lulo Restaurant': 385, 'Sugar Zoo': 386, 'Hope Sculpture By Robert Indiana': 387, 'Bodyslimdown Garcinialossfat': 388, 'Adventurers Family Entertainment Center': 389, 'Museum At Eldridge Street': 390, 'La Salle Deli': 391, 'Department of Psychiatry of Columbia University': 392, 'Futura Food': 393, 'Yue Fung Usa Enterprise Incorporated': 394, 'Coney Island Beach': 395, 'Village Community Boathouse': 396, 'Cherry Hill Fountain': 397, 'Museum Of Comic And Cartoon Art': 398, 'Museum Of American Gangster': 399, 'Minetta Triangle': 400, 'Maple Grove Park': 401, 'General Grant National Memorial': 402, 'Museum Of American Humor': 403, 'Kernan Farms': 404, 'Chanin News Corporation': 405, 'Vinmont Veteran Park': 406, 'Alcoholics Anonymous': 407, 'Cheese Unlimited': 408, 'Eckerton Hill Farm': 409, 'Unigo': 410, 'West 14th Candy Store': 411, 'Park Heenan': 412, 'Imported Foods': 413, 'Sign Language Center': 414, 'The Pearly Gates': 415, 'Machate Circle': 416, 'Pasta Resources': 417, 'Carlton Park Foster': 418, 'Gansevoort Plaza': 419, 'Barrett Park': 420, 'A Night at the Museum': 421, 'Ronnybrook Farm Dairy': 422, 'Crack Corn': 423, "Lucy's Whey": 424, 'Halal Gyro Express': 425, 'Henry Brooks Senior Housing': 426, 'Ticketmaster': 427, 'Miracle Thai': 428, 'Sustainable Table': 429, 'Pralls Island': 430, 'Chinatown Fair Family Fun Center': 431, 'Halal Cart': 432, 'Macaron Day': 433, 'Invictus Care': 434, 'Pilsudski Josef Inst of Amer': 435, 'The General Theological Seminary': 436, 'Muscota Marsh': 437, 'Historic Rose Garden': 438, 'Seton Falls Park': 439, 'Drug Treatment Centers Manhattan': 440, 'OU Israel Free Spirit': 441, 'Godiva Cafe': 442, 'East Village Cheese': 443, 'Spuyten Duyvil Shorefront Park': 444, "Harry & Ida's Meat and Supply": 445, 'Asian Street Eats': 446, 'New York Career Training & Advancement': 447, "Prison Ship Martyrs' Monument": 448, 'Huguenot Ponds Park': 449, 'Urbani Truffles': 450, "Chef's Cut": 451, 'Sugar Town': 452, 'Throgs Neck Park': 453, 'Broadway': 454, 'Care Resource Network': 455, 'Alexander Hamilton Statue': 456, 'CloveS Tail': 457, 'Rainey Park': 458, 'Nutbox': 459, 'Silver Lake Park': 460, 'Rockaway Beach Boardwalk': 461, 'Pulmonary Wellness & Rehabilitation Center': 462, 'Haven Avenue': 463, 'Cesare Olive Oil & Vinegars': 464, 'The Dunes Rehab NYC': 465, 'Pershing Square Plaza': 466, 'Medcom': 467, 'Bow Bridge': 468, 'Cuny Hunter College': 469, 'Korean War Veterans Memorial Park': 470, 'WolfeS Pond Park': 471, 'Flushing Airport': 472, 'Maison Blanche': 473, 'Nell Singer Lilac Walk': 474, 'Ny Newstand Candy and Grocery': 475, 'Vernam Barbadoes Peninsula': 476, 'USS Growler Submarine': 477, 'The Tiger Lily Kitchen': 478, 'Centflor Mfg': 479, 'Six Diamonds Park': 480, 'High Line Art': 481, 'Chocolat Moderne': 482, 'Planeview Park': 483, 'Aesop Park': 484, 'Staats Circle': 485, 'Francis Lewis Park': 486, 'Cooper Park Dog Run': 487, 'Force Basketball': 488, 'Parkslope Eatery': 489, "Faicco's Italian Specialties": 490, "Dickson's Fine Brines": 491, "Pino's Prime Meat Market": 492, 'Twenty Four Sycamores Park': 493, 'Casse Cou': 494, 'New Star Fish Market': 495, 'Fred Tea Shop': 496, 'University of Waterloo Manhattan': 497, 'St CatherineS Park': 498, 'Williamsbridge Oval Park Dog Run': 499, 'Harlem Carmiceria Hispana & Deli': 500, 'AXA Equitable Center': 501, 'Gruppo Cioccolato Internazionale': 502, 'Bow Ding Seafood': 503, 'Shore Road Park': 504, 'Cunningham Park': 505, 'Dolphin Fish Market': 506, 'Police Athletic League': 507, 'Eat Harlem': 508, "Dylan's Candy Bar": 509, 'Columbia University Medical Center Medical School': 510, 'Laredo Avenue Parcel': 511, 'Fearless Girl': 512, 'Best Buy Shellfish': 513, 'Permanent Mission of Portugal To the United Nations': 514, 'Combs Junior College': 515, 'Lincoln Center Plaza': 516, 'Meat Corporation Luis': 517, 'Permanent Mission of Cuba To the United Nations': 518, 'Washington Park JJ Byrne Playground': 519, 'ChutneyNYC Broadway Bites': 520, 'Tottenville Pool': 521, 'Juniper Valley Park': 522, 'The Jewish Museum': 523, "Hunt's Point Farmers Market": 524, 'Gopher Broke Farm': 525, 'Westside Market NYC': 526, 'Central Park Tennis Center': 527, 'Village Care of New York': 528, 'Malaysia Beef Jerky': 529, "Women's Rights Pioneers Monument": 530, 'Harlem River Park': 531, 'Albanese Rudolph Butchr': 532, 'German House': 533, 'Japan Premium Beef': 534, 'National Museum Of Hip Hop': 535, 'American Friends of Nishmat': 536, 'African Burial Ground National Monument': 537, 'Harriet Tubman Memorial Statue': 538, 'Reclining Liberty': 539, 'World Trade Center Memorial Foundation': 540, 'Park Carlton': 541, 'College Point Park': 542, 'Our Lady Queen Of Angels': 543, 'East River Waterfront Esplanade': 544, 'Museum Of Modern Art': 545, "Owen's Candy Stand": 546, 'Union Square Metronome': 547, 'American Friends of the Bible': 548, 'Shakespeare Garden': 549, 'Matthes Roland': 550, 'International Process Svc': 551, 'Sugarfina': 552, 'Gateway Newstands': 553, 'Columbia Rehabilitation and Regenerative Medicine': 554, 'Museum Of Holography': 555, 'Churro & Elote Esquite Cart': 556, 'Gimbels Traverse': 557, 'Harlem Chocolate Factory': 558, 'Major Mark Park': 559, 'Heritage Park': 560, 'Lot Surf': 561, 'NYBG Greenmarket': 562, "Hershey's Chocolate World": 563, 'Suboxone Treatment Clinic': 564, 'Dribbl Basketball': 565, 'Jacques Torres Chocolate': 566, 'John Jay Park': 567, 'Confectionery': 568, 'Salmagundi Club Museum': 569, "DiMenna Children's History Museum": 570, 'Rockaway Beach And Boardwalk': 571, 'Hyosung Park': 572, 'Arthur Ashe Stadium': 573, 'Damrosch Park': 574, 'Stars Clinic': 575, 'Laderach': 576, 'ArchCare at Mary Manning Walsh Home': 577, 'Chelsea Eats': 578, 'Blood Manor': 579, 'Project Renewal Clinton Residence': 580, 'Park Crest': 581, 'Westerleigh Park': 582, 'HunterS Point Park': 583, 'Halal Food': 584, 'Louis Valentino Jr Park & Pier': 585, 'Institute of Electrical & Electronic Eng': 586, 'Ambassador Candy Store': 587, 'West Harlem Piers': 588, 'Chocolate Covered Everything': 589, 'Arthur Brisbane Memorial': 590, 'McCarren Softball Diamond': 591, "Soldiers' & Sailors' Monument": 592, 'African Burial Ground Memorial Office': 593, 'Itamae Japanese Sushi': 594, 'Seward Park': 595, 'Touch of Elegance Events': 596, 'Nespresso': 597, 'Gapstow Bridge': 598, 'New York City Hall': 599, 'Sobel Court Park': 600, 'St Nicholas Park': 601, 'Prometheus Statue': 602, 'Allen Mall One': 603, 'American Musical And Dramatic Academy': 604, 'Citarella Gourmet Market Upper West Side': 605, 'Amedei': 606, 'Windfall Farms': 607, 'East Village Meat Market': 608, 'Say Yes To Education Teachers College': 609, 'Pretzel Time': 610, 'Boricua College': 611, 'Flushing Greens': 612, 'Pat Hoffman Friedman Playground Central Park': 613, 'Chef Recipes': 614, 'RosemaryS Playground': 615, 'NP Agency Inc': 616, 'Museum Views': 617, 'Hybrid Oak Woods Park': 618, 'Conservancy For Historic Battery Park': 619, 'Cakes Confidential': 620, 'Vladeck Park': 621, 'Golden Way Market': 622, 'Sea Breeze Fish Market': 623, 'Equity Point': 624, 'Laruleta Meat Mkt': 625, 'Mamak Cart': 626, 'Bar And Grill Park': 627, 'Cuny City College': 628, 'H & Y Grocery Store': 629, 'Stuyvesant Organic': 630, 'Montefiore Square': 631, "Bill's Lobby Stand": 632, 'Christodora House': 633, 'Felix Potin Chocolatier': 634, 'Macarthur Park': 635, 'Paerdegat Park': 636, 'Columbus Park': 637, 'Alice Austen Park': 638, "Harlem Children's Zone": 639, 'Senior Guidance': 640, 'Underwood Park': 641, 'Poseidon Bakery': 642, 'Change': 643, 'Wai Wah Market': 644, 'Elite Private Yachts New York': 645, 'Carl Schurz Big Dog Run': 646, 'Gorman Park': 647, 'American University of Barbados School of Medicine': 648, 'China Tea Herbs & Health Products': 649, 'Carniceria La Rosa': 650, "Tom's Dog Run": 651, 'Chappetto Square': 652, 'Shivkrupa': 653, 'Macombs Dam Park': 654, 'Williams Fruit Farm': 655, 'Trinity Park': 656, '5th Avenue': 657, 'Sugar and Plumm': 658, 'Queens Farm Park': 659, 'Guan Hua Candy Store': 660, 'George Washington Bridge Park': 661, 'Oldest Manhole Cover In NYC': 662, 'Laboratory Institute of Merchandising': 663, "See's Candies": 664, 'Shi Eurasia': 665, "St Lucy's Schl": 666, 'Li Lac Chocolates': 667, 'Sweet Gifts Video Cafe': 668, 'Tiffany Street Pier': 669, "82nd Street St Stephen's Greenmarket": 670, 'Schatzie the Butcher': 671, 'Gourmet Today Publctn': 672, 'Collyer Brothers Park': 673, 'Borough Of Manhattan Community College Of The City University Of New York': 674, "Marky's on Madison": 675, 'Mott St Senior Center': 676, 'Cherry Tree Park and Playground': 677, 'Coney Island Beach and Boardwalk': 678, 'Father Duffy Square': 679, 'Brust Park': 680, "Sorrentino's Market": 681, "Royce' Chocolate": 682, 'West Thames Dog Park': 683, 'Sugar Foods Corporation': 684, 'Moscow On The Hudson': 685, 'De Witt Clinton Park': 686, 'A Caring Hand': 687, 'Daniel Carter Beard Mall': 688, 'Our Heros Food Truck': 689, 'Gourmet Garage': 690, 'P S 29 Ballfield': 691, 'Reliable Corporation Lobby Newstand': 692, 'Meghna Grocery & Halal Meat': 693, 'Columbia University Inst Real Estate': 694, 'Park Joon': 695, 'Ue Enterprises': 696, 'El Ciro Fish Market': 697, 'Lt Joseph Petrosino Park': 698, 'Fresh Creek Nature Preserve': 699, "Jeff Koons' Balloon Flower Sculpture at WTC": 700, 'Drummers Grove': 701, 'Manhattan By Sail': 702, 'Addiction Recovery Hotline Manhattan': 703, 'Tepper Triangle': 704, 'Queens Botanical Garden Farmers Market': 705, 'Battery Park City Tennis': 706, 'National Race Masters': 707, 'Marble Hill Playground': 708, 'Columbus Gourmet Food': 709, 'L J Campbell Walter': 710, 'Harvard Business School Press': 711, 'Meat On A Stick': 712, 'Dyckman Street Viaduct': 713, 'Drew University U N Program': 714, 'NYC Department of Parks': 715, 'Asia Society and Museum': 716, 'Black Top Street Hockey': 717, 'Delimanjoo': 718, 'Met Fresh': 719, 'Lewis Home Care Agency': 720, 'Winterdale Arch': 721, 'Halal Street Meat Cart': 722, 'Brooklyn Academy Of Music': 723, 'Great Kills Park Hillcrest': 724, 'Riverside Park South': 725, 'Museum Of Mathematics': 726, 'JLA Meat Market': 727, 'Fresh&co.': 728, 'Park Vanderbilt': 729, 'Evergreen Park': 730, "Gilda's Club New York City": 731, 'LGBT Memorial': 732, "St Mark's Church Greenmarket": 733, 'Brooklyn Bridge Park Pier 5': 734, 'Firefighters Memorial Park At Ritz Plaza': 735, 'Truce Garden': 736, 'Kings College': 737, 'Pier Sixty New York City Event and Weddings Venue': 738, 'Rehab Helpline': 739, 'Midtown Catch': 740, 'South Slope Dog Run': 741, 'Einstein Mural by Kobra': 742, 'Slope Park': 743, 'Clason Point Park': 744, 'C & B Candy Store': 745, 'City of New York': 746, 'Korean War Veterans Memorial': 747, 'Givans Creek Woods': 748, 'William F Passannante Ballfield': 749, 'Sweet Town': 750, 'Bryant Hill Garden': 751, 'Abingdon Square': 752, 'Hot Soup Cart': 753, 'Indie Fresh': 754, 'Thomas J Cuite Park': 755, 'Pulaski Park': 756, 'Prezant Sonia': 757, 'Virginia Park': 758, "Christie's Sculpture Garden": 759, 'City Market Cafe': 760, 'Daggett Museum For Living Artists': 761, 'G B A Meats': 762, 'Papabubble': 763, 'Eatside': 764, 'East Side Meat Corporation': 765, 'Leonidas Fresh Belgian Chocolates': 766, 'Arsenal North': 767, 'RAIN Inwood Neighborhood Senior Center': 768, 'Jazz Museum': 769, 'Amersfort Park': 770, 'Inspir Carnegie Hill': 771, 'Lincoln Square': 772, 'S & A 96 Candy Store': 773, 'Equity Park': 774, 'Halal Food Cart': 775, 'Waterfalls in Central Park': 776, 'Association of Art Museum Directors': 777, 'Kissena Park': 778, 'Raoul Wallenberg Forest': 779, 'Share The Care': 780, 'Are East River Science Park': 781, 'Southwest Recreation Center': 782, 'Magness Arena': 783, '9th Street Historic Park': 784, 'Exceptional Voice': 785, 'Invisible Museum': 786, 'Safe Harbor Home Care': 787, 'Blitz Bowl Football Bowling': 788, 'Amino Kit': 789, 'Rocky Mountain Arsenal': 790, 'Wabash Trailhead': 791, 'The Denver Hospice': 792, 'La Familia Recreation Center': 793, 'City Park': 794, 'Doors Open Denver': 795, 'Martinez Joseph P Park': 796, 'Denver Museum Of Nature & Science': 797, 'Hirshorn Park': 798, 'Mimi': 799, 'Troy Chavez Memorial Peace Garden': 800, 'Capitol Hill Reservoir': 801, 'O Sullivan Art Museum': 802, 'Transition & Recovery Counseling Center': 803, 'Bradley': 804, 'Vanderbilt Park': 805, 'Commons Park': 806, 'University Park': 807, 'Montbello Open Space': 808, 'Elitch Gardens': 809, 'Rocky Mountain Lake Park': 810, 'Russell Square Park': 811, 'Pferdesteller Park': 812, 'Worldwide Adult DayCare': 813, 'Ciancio Park': 814, 'Boyd J Langston Park': 815, 'Lindsley Henry S Park': 816, 'Chavez Cesar E Park': 817, 'Hampden Heights Park': 818, 'University Of Colorado Museum Of Natural History': 819, 'Hutchinson Park': 820, 'Essential History Expeditions': 821, 'Hallett Fund Academy': 822, 'Swansea': 823, 'City Of Takayama Park': 824, 'College View': 825, 'Denver Technological Center': 826, 'Coronado Pkwy': 827, "The Children's Museum of Denver at Marsico Campus": 828, 'Cherry Creek Greenway': 829, 'Centennial Park': 830, 'Empower Field at Mile High': 831, 'City Of Chennai Park': 832, 'Hallack Park': 833, 'Smith': 834, 'Central Park': 835, 'W H Ferguson Park': 836, 'Jefferson Square Park': 837, 'The Courtyards at Mountain View': 838, 'Ashley': 839, 'Swansea Neighborhood Park': 840, 'Highland Park': 841, 'Edison': 842, 'Meow Wolf Denver Convergence Station': 843, 'Munroe': 844, 'Wilis Case Golf Course': 845, "Denver Men's Hockey": 846, 'Eastmoor Park': 847, 'Sanchez Paco Park': 848, 'Shadow Array': 849, 'Quadrivium': 850, 'Liberty Day': 851, 'Quality Hill Park': 852, 'United Drug Rehab Group Denver': 853, 'Barnum': 854, 'Manley James N Park': 855, 'Lowry Open Space': 856, 'Ashford University': 857, 'Wow Interactions': 858, 'Badging Office': 859, 'Community College Of Aurora': 860, 'Flourish Supportive Living': 861, 'Pinecrest Village Park': 862, 'Morrison George Sr Park': 863, 'Mile High Youth Corps': 864, 'Hungarian Freedom Park': 865, 'International Neural Renewal': 866, 'Jefferson Park': 867, 'Fairview': 868, "Outpatient Women's Treatment Services": 869, 'Hampden Heights North': 870, 'Force': 871, 'Housing & Urban Development': 872, 'Quick Lenore B Park': 873, 'Gust': 874, 'Garfield Lake Park': 875, 'Five Points Historic Cultural District': 876, 'Museum Of Computer Technology': 877, 'Wallace Park': 878, 'Douglass Fredrick Park': 879, 'Frederick Douglass Park': 880, 'Westwood Park': 881, 'Doull': 882, 'Slavens': 883, 'Denver Rehabilitation Clinic': 884, 'Overland Pond Park': 885, 'Whittier': 886, "DU's historic Chamberlin Observatory": 887, 'Red Light Camera 6th & Lincoln': 888, 'Amesse': 889, 'VR Arcade': 890, 'Museum Of Contemporary Art Denver': 891, 'Lowry': 892, 'Chaffee Park': 893, 'Kavod Senior Life': 894, 'Zion Learning Center': 895, 'Articulated Wall by Herbert Bayer': 896, 'James Manley Park': 897, 'Observatory Park': 898, 'Marston Lake Walking Trail': 899, 'Oakland': 900, 'Dayspring Villa': 901, 'Wildlife World Museum': 902, 'City Of Brest Park': 903, 'Palmer': 904, 'Cherry Creek Recreation Area': 905, 'Divine Hospice & Palliative Care': 906, 'US Customs House': 907, 'City Of Kunming Park': 908, 'KaboomVR': 909, 'St PatrickS Neighborhood Park': 910, 'Sanderson Gulch Park & Trail': 911, 'Central Park Stapleton': 912, 'Scottish Angus Cow and Calf': 913, 'High Line Canal Greenway': 914, 'Barnum Dog Park': 915, 'Denver Kickball Coalition': 916, 'Falcon Park': 917, 'National Addiction Rehabs Denver': 918, 'Coors Field': 919, 'F 15 Park': 920, 'Byers Evans House Museum': 921, "University Of Denver Pioneers Men's Lacrosse": 922, 'Gates Crescent Park': 923, 'Dcpa Sculpture Park': 924, 'Decatur West Personal Care': 925, 'Marycrest Assisted Living': 926, 'Forney Museum': 927, 'Lil Patch of Heaven': 928, 'Washington Park': 929, 'Eisenhower Mamie Doud Park': 930, 'Colorado Tennis Association': 931, 'High Line East': 932, 'Swansea Park': 933, 'Bear Creek Park': 934, 'Coral Sphere': 935, 'Veterans Field': 936, 'United States Mint': 937, 'Ruby Hill Park': 938, 'Kaboom Virtual Reality Arcade': 939, 'Square Park': 940, 'Skyline Park': 941, 'Mestizo Curtis Park': 942, 'Museum Of Anthropology': 943, 'Babi Yar Park': 944, 'Porter Wound Care Center': 945, 'Centennial Gardens': 946, 'Cranmer Park': 947, 'Centura Rehabilitation': 948, 'Haven of Care Assisted Living': 949, 'Epoch Assisted Living of Denver': 950, 'Viking Park': 951, 'It Professional': 952, 'Kunsmiller': 953, 'Zuni Park': 954, 'Crowley Boats': 955, 'Creekfront Park': 956, 'Denver Phlebotomy School': 957, 'Denver Art Museum': 958, 'Dr Gene Mahaffey': 959, 'Golden Triangle': 960, 'Operation School Bell': 961, 'Macintosh Kenneth M Park': 962, 'Colorado Film School': 963, 'Highlands Integrative Pediatrics': 964, 'Northside Park': 965, 'Senior Care Authority': 966, 'Heritage Club At Denver Tech Center': 967, 'Inspiration Point Park': 968, 'Bryant Webster': 969, 'Balfour': 970, 'Carla Madison Rec Center': 971, 'Newlon': 972, 'Martin Luther King Jr Park': 973, 'University Of Denver': 974, 'Attorney General': 975, 'Stapleton Development Corporation': 976, 'Flores Hector M Park': 977, 'City Of Cuernavaca Park': 978, 'Speer Blvd Park': 979, 'Mcmeen': 980, 'Schenck': 981, 'Drug Detox Centers Denver': 982, 'Sail Park': 983, 'Walker Madame Cj Park': 984, 'Carla Madison Dog Park': 985, 'Congress Park': 986, 'Ford Barney L Park': 987, 'Highland Gateway Park': 988, 'Knapp': 989, 'Gates Planetarium': 990, 'Tzun Tzu Military History Museum': 991, 'Alamo Placita Park': 992, 'Sullivan Gateway': 993, 'Brookdale Denver Tech Center': 994, 'Uplands Park': 995, 'City View Park': 996, 'Wartburg College West': 997, 'Odyssey': 998, 'Pelham College': 999, 'Athmar Park': 1000, 'Colorado State University': 1001, 'Rosemark At Mayfair Park': 1002, 'Parkfield': 1003, 'Friends of Historic Ft Logan': 1004, 'Grant Frontier Park': 1005, 'New Century Hospice': 1006, 'Ash Grove Park': 1007, 'Johnson': 1008, 'Silverman Melvin F Park': 1009, 'Wyman Multiple Path Iii': 1010, 'Mca': 1011, 'Steck': 1012, 'Shiki Dreams': 1013, 'University of Colorado At Denver and Health Sciences Center': 1014, 'Feel Good Llc': 1015, 'Grant Ranch Village Center': 1016, 'Wallace George M Park': 1017, 'Colorado Democratic Party': 1018, 'Garland Street Park': 1019, 'McPhilomy Commercial Products': 1020, 'Conservatory Green': 1021, 'Cheltenham': 1022, 'Fairmont': 1023, 'CommonGround Golf Course': 1024, 'Belleview Suites at DTC': 1025, 'Mile High Continuing Care': 1026, 'Hutchinson Theodore A Park': 1027, 'National American University': 1028, 'Dreiling Ruth Lucille Park': 1029, 'Rosamond Park': 1030, 'Archuleta': 1031, 'Denver School Of Nursing': 1032, 'Fuller Park': 1033, 'Washington Park Boathouse': 1034, 'Riverfront Park': 1035, 'Balfour Riverfront Park': 1036, 'Independence Plaza Parking': 1037, 'Magna Carta Park': 1038, 'Holm': 1039, 'Cheesman Park': 1040, 'Western Motorsports': 1041, 'Teller': 1042, 'St John Vianney Theological Seminary': 1043, 'Veterans Park': 1044, 'Kittredge Park': 1045, 'Frances Wisebart Jacobs Park': 1046, 'Cu Art Museum': 1047, 'Harvard Gulch West': 1048, 'Westerly Creek': 1049, 'The Money Museum': 1050, 'Millennium Bridge': 1051, 'Village Greens Park': 1052, 'InnovAge Headquarters': 1053, 'University College': 1054, 'Elaine T Valente Open Space': 1055, 'Cuatro Vientos Four Winds Park': 1056, 'Molly Brown House Museum': 1057, 'Ball arena formerly Pepsi Center': 1058, 'Downtown Childrens Playground': 1059, 'Clyfford Still Museum': 1060, 'Valdez': 1061, 'Goldrick': 1062, 'French & Associates': 1063, 'Aviation & Space Center of the Rockies': 1064, 'The Yearling': 1065, 'D C Burns Park': 1066, 'Travel Center Specialists': 1067, 'Crating Technologies': 1068, 'Jewels of Highlands House Tour': 1069, 'Rocky Mountain Arsenal National Wildlife Refuge': 1070, 'Spectrum Retirement Communities': 1071, 'The Gardens At St Elizabeth': 1072, 'Williams Dr Daniel Hale Park': 1073, 'Cook Park Recreation Center': 1074, 'VR Visage': 1075, 'Marion Street Community Center': 1076, 'Barrett': 1077, 'Technical Information Associates': 1078, 'MacKenzie House': 1079, 'Denver Drug Rehab': 1080, 'Confluence East Park': 1081, 'Village Place Park': 1082, 'Sabin': 1083, 'City Of Potenza Park': 1084, 'Bonnie Brae Park': 1085, 'Colorado Access': 1086, 'Carson': 1087, 'Lightrail Plaza': 1088, 'Columbian': 1089, 'Samuels': 1090, 'Colorado Heights University': 1091, 'Northfield Athletic Complex': 1092, 'Assited Living Denver': 1093, 'City Of Axum Park': 1094, 'Denver Firefighters Museum': 1095, 'Colorado Military History Museum': 1096, 'Sonny Lawson Park': 1097, "University Of Denver Pioneers Men's Basketball": 1098, 'Healing Lifes Pains': 1099, 'Mitchell': 1100, 'X Gaming': 1101, 'Rotella Park': 1102, 'Yeshiva Toras Chaim': 1103, 'Curtis Park': 1104, 'All the Comfort of Home': 1105, 'Martin Luther King Park': 1106, 'Alcohol Drug Rehab Denver': 1107, 'Bromwell': 1108, 'Marrama': 1109, 'Benedict Fountain Park': 1110, 'Colorado Home Care': 1111, 'Sobriety House': 1112, 'Traylor': 1113, 'Cherry Creek Gun Club': 1114, 'Cycleton Lowry': 1115, 'East 86Th Ave Trail Greenway': 1116, 'Berkeley Leashless Dog Park': 1117, 'Sanderson Gulch Irving and Java': 1118, 'Garrison And Union Park': 1119, 'Education Commission of the States': 1120, 'James H Platt Park': 1121, 'Metro Park': 1122, 'Montage Heights': 1123, 'Bearing Institute': 1124, 'Green Valley Ranch East Park': 1125, 'Castro': 1126, 'Army National Guard Recruiting': 1127, 'Dry Gulch Trail': 1128, 'Cook Judge Joseph E Park': 1129, 'Steele Street Park': 1130, 'Independent Growth Home Healthcare Agency': 1131, 'Seymour Deborah J Psyd': 1132, 'Pulaski Park and Playground': 1133, 'Crestmoor Park': 1134, 'Independence House Northside': 1135, 'Wooden Open': 1136, 'InnovAge PACE Denver': 1137, 'Bayaud Park': 1138, 'Daniels and Fisher Tower': 1139, "Sloan's Lake Park": 1140, 'Harvard Gulch Golf Course': 1141, 'Pacific College of Allied Health': 1142, 'Senior Housing Options The Barth Hotel': 1143, 'Valverde Park': 1144, 'PasquinelS Landing': 1145, 'Montbello Civic Center Park': 1146, 'Denver Roller Derby': 1147, 'Pinnacle City Park Hoa': 1148, 'Ashland Recreation Center': 1149, 'Community Plaza Park': 1150, 'Weir Gulch Marina Park': 1151, 'New Freedom Park': 1152, 'Cottage Hill Senior Apts': 1153, 'Lilu Star Indoor Playground': 1154, 'Adolescent Psychiatry & Pediatric Psychiatry': 1155, 'Tree Sculpture': 1156, 'Wings Over The Rockies Air & Space Museum': 1157, 'Niver Creek Trail Greenway': 1158, 'Denison Park': 1159, 'Denver City And County Government Museums': 1160, 'Colfax': 1161, 'City Of Karmiel Park': 1162, 'Harvey Park': 1163, 'Wheel Fun Rentals Berkeley Park': 1164, 'Burns Dc Park': 1165, 'Harvard Gulch Park': 1166, 'La Alma Lincoln Park': 1167, 'Altura Park': 1168, 'Southmoor Park': 1169, 'Siemann Educational': 1170, 'Nurse Family Partnership': 1171, 'Dailey Park': 1172, 'Ellis': 1173, 'Cherry Creek Park': 1174, 'Habitat Park': 1175, 'Overland Park Golf Course': 1176, 'Molly Brown Summer House': 1177, 'Aspen University': 1178, "Children's Hospital Colorado KidStreet": 1179, 'Distortions Monster World': 1180, 'Johnson Habitat Park': 1181, 'Denver Outdoor Adventure Company': 1182, 'Aztlan Park': 1183, 'Lake Of Lakes Park': 1184, 'Franco Bernabe Indio Park': 1185, 'Ruby Hill': 1186, 'Mckinley Thatcher': 1187, 'Hentzel Paul A Park': 1188, 'The Great Lawn Park': 1189, 'Fred Thomas Park': 1190, 'Colorado Museum Of Natural History': 1191, 'Sandstone Care': 1192, 'Vintage Motos Museum': 1193, 'Colorado Parks & Wildlife': 1194, 'Serenity House Assisted Living Office': 1195, 'Emeritus at Highline': 1196, 'Heart 2 Heart 4 Seniors': 1197, 'Garland David T Park': 1198, 'Sandhofer Reservoir': 1199, 'Central Park Rec Center': 1200, 'Aspgren Clifford Park': 1201, 'Welch Park': 1202, 'Kelly Open Space': 1203, 'American Museum Of Western Art': 1204, 'Sunken Gardens Park': 1205, 'Avalanche Ice Girls': 1206, 'MGA Crisis Intervention': 1207, 'VNA Hospice': 1208, 'The Manor on Marston Lake': 1209, 'Emeritus At Pinehurst Park': 1210, 'Jacobs Frances Weisbart Park': 1211, 'Montclair Recreation Center': 1212, 'Mcclain Thomas Ernest Park': 1213, 'Eagleton': 1214, 'Lincoln': 1215, 'Westerly Creek Park': 1216, 'Garden Place': 1217, 'Mcglone': 1218, 'Denison': 1219, 'Sandos M L Sam Park': 1220, 'Adventist University Of Health Sciences Denver Campus': 1221, 'Sanderson Gulch': 1222, 'Park Hill': 1223, 'Colorado Rockies Baseball Club': 1224, 'Roberts': 1225, 'Little Boxcar Dog Park': 1226, 'Bible James A Park': 1227, 'Denver Housing Authority': 1228, 'Eastern Star Masonic Retirement Community': 1229, 'Southwest Auto Park': 1230, 'Lueck Open Space': 1231, 'Urban Assault Ride by New Belgium': 1232, 'Pinehurst Park': 1233, 'Robert H McWilliams Park': 1234, 'City Of Ulaanbaatar Park': 1235, 'Montclair': 1236, 'CenturyLink Tower': 1237, 'Greenway Park': 1238, 'Place': 1239, 'Emeritus at Roslyn': 1240, 'National Ballpark Museum': 1241, 'Huston Lake Park': 1242, 'SloanS Lake Park': 1243, 'Paco Sanchez Park': 1244, 'DC Stop Smoking': 1245, 'Eunique Associates': 1246, 'Berkeley Lake Park': 1247, 'Fishback Park': 1248, 'Hunger Intervention Program': 1249, 'Sub Zero Quality Prime Repair': 1250, 'Arctic Viking Experts': 1251, 'Western Safety Products Inc': 1252, 'Seattle Metropolitan Police Museum': 1253, 'Counterbalance Park': 1254, 'Rainier Beach Urban Farm and Wl': 1255, "Edward's Refrigerator Repair Service": 1256, 'Ward Springs Park': 1257, 'Gas Works Park': 1258, 'Twelfth Avenue South Viewpoint': 1259, 'Miller Triangle': 1260, 'Seattle Sub Zero Repair': 1261, "Alex's Watch & Clock Repair": 1262, 'Lower Woodland Park Playfields': 1263, 'Jefferson Park Golf Course': 1264, 'Little Brook Park': 1265, 'View Ridge Es Playfield': 1266, 'Mount Baker Boulevard': 1267, 'American Firefighter': 1268, 'Kingfisher Natural Area On Tc': 1269, 'Homewood Park': 1270, 'Puget Creek Natural Area': 1271, 'Henry Art Gallery': 1272, 'Bellevue Place': 1273, 'Hubbard Homestead Park': 1274, 'St Marks Greenbelt': 1275, 'Beaver Pond Natural Area On Tc': 1276, 'Horiuchi Park': 1277, 'Arrowhead Gardens': 1278, 'Roanoke Park Watch Repair': 1279, 'Maple Leaf Reservoir Park': 1280, 'Superb Custom Tailors': 1281, 'Pike Place Market Gum Wall': 1282, 'The Watch & Jewelry Repair': 1283, 'Western Avenue Senior Housing': 1284, 'Washington Huskies Football': 1285, 'GameWorks': 1286, 'Rubber Chicken Museum': 1287, 'Washington Care Center': 1288, 'Haller Lake Community St End': 1289, 'Court Furniture Rental': 1290, 'East Portal Viewpoint': 1291, 'Virgil Flaim Park': 1292, 'Ballard Sails': 1293, 'Alvin Larkins Park': 1294, 'Bell Street Park Boulevard': 1295, 'Darren Sub Zero Repair Master': 1296, 'Downtown Seattle Masters': 1297, "Hat n' Boots": 1298, 'Metro Tailoring & Alterations': 1299, 'Dr Jose Rizal Park': 1300, 'Luna Park Hoa': 1301, 'Lakewood Moorage': 1302, "Laura's Alteration & Tailor Express": 1303, 'Naval Reserve Armory': 1304, 'Matthews Beach Park': 1305, 'Lake City Young People': 1306, 'Gerberding Hall': 1307, 'Mr. Appliance': 1308, 'Northwest Blind Repair': 1309, 'Pioneer Square': 1310, 'Smith Cove': 1311, 'Pritchett Island Beach': 1312, 'Bhy Kracke Park': 1313, 'Andover Place': 1314, 'Othello Playground': 1315, 'Sole Repair': 1316, 'Reel Grrls': 1317, 'Homer Harris Park': 1318, 'Katie Blacks Garden': 1319, 'Ridge Custom Upholstery': 1320, 'Nantes Park': 1321, 'South Ship Canal Trail': 1322, 'Cheasty Greenspace': 1323, 'Spada Homes': 1324, 'Spruce Street Mini Park': 1325, 'National Oceanic and Atmospheric': 1326, 'Klondike Gold Rush National Historical Park Seattle Unit': 1327, 'Froula Playground': 1328, 'Marvins Garden': 1329, 'Fred Lind Manor': 1330, 'Queen Anne Upholstery and Refinishing': 1331, 'Judge Charles M Stokes Overlk': 1332, 'Baker Park On Crown Hill': 1333, 'Soundview Playfield': 1334, 'Rider Oasis': 1335, 'Seattle Parks and Recreation': 1336, 'Death Museum Store': 1337, 'Crown Hill Park': 1338, 'Refrigerator Repair': 1339, 'Spokane Street Bridge': 1340, 'Seacrest Park': 1341, 'Central Seattle Panel of Consultants Inc': 1342, 'Boylston Place': 1343, 'Interbay Golf': 1344, 'European Senior Care': 1345, 'Upholstery Outfitters of Seattle': 1346, 'Seattle Appliance Specialist': 1347, 'Arroyos Natural Area': 1348, 'Coast Guard Museum': 1349, 'Thrive Center For Healing Traditions': 1350, 'Kirke Park': 1351, 'The Prince From Minneapolis Exhibit': 1352, 'Grady KitchenAid Repair Master': 1353, 'Antioch University Seattle': 1354, 'A & J Commissary': 1355, 'Wrench Bicycle Workshop': 1356, 'Thermador Appliance Mavens': 1357, 'Volunteer Park Conservatory': 1358, 'The Bicycle Repair Shop': 1359, 'Northgate Plaza': 1360, 'Connections': 1361, 'Delridge Playfield': 1362, 'Bryant Neighborhood Playground': 1363, 'Atlantic City Boat Ramp': 1364, 'Washington Trust For Historic Preservation': 1365, 'Jack Block Park': 1366, 'Thermador Repair Group Industrial Disctrict': 1367, 'Highland Drive Parkway': 1368, 'Fairview Walkway': 1369, 'New Mark Tailor': 1370, 'Budget Blinds of West Seattle & Central South': 1371, 'Linden Orchard Park': 1372, 'Helene Madison Pool': 1373, 'Interbay P Patch': 1374, 'Walrus Machine': 1375, 'Ercolini Park': 1376, 'Seattle Drug Rehab': 1377, 'Wings Over Washington': 1378, 'Montlake Playfield': 1379, 'Lake City Watch Repair': 1380, 'University House Wallingford': 1381, 'Buck Creek Presbyterian Camp': 1382, 'Sandel Playground': 1383, 'Wallingford Playfield': 1384, 'Als Music Video & Games': 1385, 'Schmitz Preserve Park': 1386, 'York Playground': 1387, 'Bar S Playground': 1388, 'Rose Garden at Woodland Park Zoo': 1389, 'Kerry Park': 1390, "Can't Look Away The Lure of Horror Film Exhibit": 1391, 'American Civil Liberties Union': 1392, 'Fremont Canal Park': 1393, 'Mount Baker Park': 1394, 'Museum Of Communications': 1395, 'College Street Park': 1396, 'Pipers Creek Natural Area': 1397, 'Appliance Masters In Seattle': 1398, 'Fuller Theological Seminary': 1399, 'Columbia Park': 1400, 'Jimi Hendrix Statue': 1401, "Raul's Viking Repair Services": 1402, 'Me Kwa Mooks Park': 1403, 'Amusement Services': 1404, 'Pearly Jones Home': 1405, 'Ne 60Th Street Park': 1406, 'Thornton Creek Natural Area': 1407, 'Nordic Museum': 1408, "Seattle Children's Museum": 1409, "Edith Macefield's House": 1410, 'Rainbow Point': 1411, 'Roanoke Street Mini Park': 1412, 'East Madison Street Ferry Dock': 1413, 'UW Medicine Diabetes Institute': 1414, 'Kitchen Repair Specialists': 1415, 'Logan Field': 1416, 'Kerry Park Franklin Place': 1417, 'Thrifty Janitorial': 1418, 'Green Lake Small Craft Center': 1419, 'Pot Business Rules For Washington': 1420, 'Kitchen Masters Specialist': 1421, 'Envision Response': 1422, 'Boren Park': 1423, 'Rainier Valley Historical Society': 1424, 'Thomas C Wales Park': 1425, 'Tashkent Park': 1426, 'Maple Wood Playfield': 1427, 'Lincoln Park': 1428, 'First Hill Park': 1429, 'Madrona Playground': 1430, 'Sacred Garden Healing Center': 1431, 'Smith Cove Park': 1432, 'Pivot Art Culture': 1433, 'Merrill Gardens at Queen Anne': 1434, 'Solo Appliance Repair': 1435, 'Elenvilla Alterations & Tailoring': 1436, 'Dragonfly Pavilion': 1437, 'Powell Barnett Park': 1438, 'Terminal 18 Park': 1439, 'Athena University': 1440, 'Magnolia Greenbelt': 1441, 'Espresso Technolgies': 1442, 'Union Bay Natural Area': 1443, 'Hiram M Chittenden Locks': 1444, 'East Queen Anne Playground': 1445, 'Dearborn Park': 1446, 'Kiwanis Ravine Overlook': 1447, 'Fairview Park': 1448, 'Meril Gardens': 1449, 'John C Little Sr Park': 1450, 'Central Vacuum Service': 1451, 'Kilbourne Park': 1452, 'Filipino American National Historical Society': 1453, 'All About Seniors': 1454, 'Licorice Fern Na On Tc': 1455, 'Rainier Beach Playfield': 1456, 'Pinehurst Pocket Park': 1457, 'Campana Quetzal': 1458, 'South Passage Point Park': 1459, 'Greenwood Park': 1460, 'West Central Grounds Maint': 1461, 'Sutherland Cooktop Repair': 1462, 'Good Shepherd Center': 1463, 'Pratt Park': 1464, 'Westcrest Park': 1465, 'TrollS Knoll': 1466, 'Amira Acne Solutions': 1467, 'Waterfall Garden': 1468, 'Myrtle Edwards Park': 1469, 'Seattle Tower': 1470, 'The Langland House Adult Family Home': 1471, 'Lamber Thermador Range Repair Service': 1472, 'Grindline': 1473, 'The Seattle Great Wheel': 1474, 'West Magnolia Playfield': 1475, 'Judkins Park And Playfield': 1476, 'Queen Ann Drive Bridge': 1477, 'Florence of Seattle': 1478, 'Uysal Alterations': 1479, 'Gemenskap Park': 1480, 'Marine Corps Recruiting': 1481, 'Ravenna Woods': 1482, 'Lavilla Meadows Na On Tc': 1483, 'Hamilton Viewpoint Park': 1484, 'Monkey Fist Marine': 1485, 'WorldBest Seattle Viking Range Repair': 1486, 'Pendleton Miller Playfield': 1487, 'Washington Park Japanese Gdn': 1488, 'Burke Museum Of Natural History And Culture': 1489, 'Solstice Park': 1490, 'Nathan Hale Stadium Complex': 1491, 'American Cultural Exchange': 1492, 'Spring Manor': 1493, 'Seattle The Top Repair': 1494, 'Fisher Pavillion': 1495, 'Thomas Street Mini Park': 1496, "Stonepath Logistics Int'l Svcs": 1497, 'Ballard Swimming Pool': 1498, 'Boat Electric': 1499, 'Martha Washington Park': 1500, 'Laurelhurst Beach Club': 1501, 'Belltown Cottage Park': 1502, 'Jenny Manes Premier Bridal Design & Alterations': 1503, 'Cinderella Tailors': 1504, 'Village Shoe Repair': 1505, 'Arbor Heights Elder Care': 1506, 'Espresso Repair Experts': 1507, 'Frink Park': 1508, 'Space Needle': 1509, 'Steamworks Espresso Service': 1510, 'Arroyo Heights': 1511, 'University Lake Shore Place': 1512, 'Elite Vend': 1513, 'Bitter Lake Playfield': 1514, 'Hing Hay Park': 1515, 'Thorndyke Park': 1516, 'Schmitz Boulevard': 1517, 'Thomas Street Gardens': 1518, 'JP Custom Upolstery and Repair': 1519, 'University Heights': 1520, "Diane's Alterations and Tailoring": 1521, 'Maple School Ravine': 1522, 'They Shall Walk Museum': 1523, 'Licton Springs Park': 1524, 'Dick Paetzke Creative Directions': 1525, 'Your Choice Appliance Service': 1526, 'Stanley Sayres Memorial Park': 1527, 'Coffee Physics': 1528, 'Thermador Repair Group Seattle': 1529, 'Queen Anne Swimming Pool': 1530, 'A Leg Up': 1531, 'United Machine Shop Service': 1532, 'Cesar Chavez Park': 1533, 'Pathfinder': 1534, 'Pacific Northwest Museum Of Motorcycling': 1535, 'Pacific Maritime Institute': 1536, 'Bridge Gear Park': 1537, 'Lake City Mini Park': 1538, 'Lake Union Yacht Center': 1539, 'NW Caravan Treasures': 1540, 'Zen GE Monogram Repair': 1541, 'Living Computer Museum': 1542, 'City University Of Seattle': 1543, 'Leventhorp Behavioral': 1544, 'ARS Appliance Repair Service': 1545, 'Webster Park': 1546, 'Always Perfect Yacht Interiors': 1547, 'Navy Recruiting District Seattle': 1548, 'Sunnyside Ave N Boat Ramp': 1549, 'Seattle Municipal Tower': 1550, 'Classic Yacht Association': 1551, 'ABC Appliance Service': 1552, 'Lakewood Playfield': 1553, 'Polar Bear Mechanical': 1554, 'Kinnear Park': 1555, 'Marshall Park': 1556, "Terry's Thermador Appliance Experts": 1557, 'Volunteer Parkway': 1558, 'University of Washington Virology Research Clinic': 1559, 'Hilltop Appliance Repair': 1560, 'Waterway': 1561, 'The Boat Rental': 1562, 'The ED Clinic': 1563, 'Malatesta Woodworks': 1564, 'Chinese Community Bulletin Board': 1565, 'Red Square': 1566, 'Emeritus at West Seattle': 1567, 'A B Ernst Park': 1568, 'Fremont Appliance Refit': 1569, 'Nathan Hale Playfield': 1570, 'American Freestyle Alterations': 1571, 'View Ridge Playfield': 1572, 'Giant Shoe Museum': 1573, 'Pioneer Association of the State of Washington': 1574, "Eidem's Custom Upholstery": 1575, 'E C Hughes Playground': 1576, 'Greenwood Phinney Ruv': 1577, 'Wisteria Park': 1578, 'Race Recycling and Compacting Equipment': 1579, 'Boyer Thermador Appliance Repair': 1580, 'Aquarist World': 1581, 'Adam Tailoring & Alterations': 1582, 'Georgetown Powerplant Museum': 1583, "Weiskind's Appliance": 1584, 'Maximum Performance Hydraulics': 1585, 'Northgate Park': 1586, 'Tanishas Hood Drag Race': 1587, 'Cleveland Playfield': 1588, "Seattle Children's Playgarden": 1589, 'Plymouth Pillars Park': 1590, 'Wallingford Steps': 1591, 'Seattle Public Schools': 1592, 'Emerald Harbor Marine': 1593, 'Seattle Interactive Media Museum': 1594, 'Cal Anderson Park and Bobby Morris Playfield': 1595, 'Westlake Park': 1596, 'Westlake Square': 1597, 'Woodland Park Off Leash Area': 1598, 'Bedford Sub Zero Appliance Repair': 1599, 'Greenwood Triangle': 1600, 'B F Day Playground': 1601, "Green Lake Pitch n' Putt": 1602, 'Broadway Hill Park': 1603, 'Merrill Gardens at First Hill': 1604, "Tim's DCS Repair Service": 1605, 'South Bark Dog Park': 1606, "Expeditors Int'l of Washington": 1607, 'Foster Island Wash Park': 1608, 'McIntyre Organ Repair Service': 1609, 'Creative Date': 1610, 'The Laser Dome': 1611, 'Home Espresso Repair and Cafe': 1612, 'Olympic Hills Es Playfield': 1613, 'West Queen Anne Playfield': 1614, 'Magnolia Care': 1615, 'Simply Frames & Miner Gallery': 1616, 'University House Resident Council': 1617, 'T Mobile Park': 1618, 'Casa Bella Home Care Services': 1619, 'Columbia Tower Club': 1620, 'Don Armeni Boat Ramp': 1621, 'Blue Dog Pond': 1622, 'Seattle Selfie Museum': 1623, 'Cal Anderson Park': 1624, 'Carleton Highlands': 1625, 'Westlake Greenbelt': 1626, 'A Place for Mom': 1627, 'Ballard Corners Park': 1628, 'Stevens Triangle': 1629, 'Center For Autism Rehabilitation & Evaluation': 1630, 'Pike Place Urban Garden': 1631, 'High Rated Repair': 1632, 'Presbyterian Retirement Communities Northwest': 1633, 'Discovery Park': 1634, 'Seattle Metro Appliance Repair': 1635, 'Sky View Observatory': 1636, 'Noras Woods': 1637, 'Dr Blanche Lavizzo Park': 1638, 'Peoples Computer Museum': 1639, 'GE Monogram Appliance Repair Service': 1640, 'Northlake Park': 1641, 'Alki Point Lighthouse Tours': 1642, 'Puget Sound': 1643, 'Demonstration Garden': 1644, 'North Beach Es Playfield': 1645, 'Champion Small Engine Repair': 1646, 'Orchard Street Ravine': 1647, 'Beacon Hill Playground': 1648, 'Meadowbrook Playfield': 1649, 'Washington Park Arboretum': 1650, 'Swansons Shoe Repair': 1651, 'Magnolia Park': 1652, 'Me Kwa Mooks Natural Area': 1653, 'Waterfront Park': 1654, 'Seattle Asian Art Museum': 1655, 'EcoGuard Appliance': 1656, 'Burke Gilman Trail': 1657, 'Bradner Gardens Park': 1658, 'Judkins Park P Patch': 1659, 'Seattle Tacoma Appliance Repair': 1660, 'Chinook Beach Park': 1661, "Jim's Cobbler Shoppe": 1662, 'PAR3 Communications': 1663, "Martha's Garden": 1664, 'Busy Shoes Quality Shoe Repair': 1665, 'College Street Ravine': 1666, 'HomeWell Senior Care': 1667, 'Drug Rehab and Sober Living Seattle': 1668, 'Lakeridge Playfield': 1669, 'Era Living': 1670, 'Pleasant Adult Family Home': 1671, 'Bitter Lake Open Space Park': 1672, 'Open Water Park': 1673, 'Prentis I Frazier Park': 1674, 'Matthews Beach': 1675, 'Twelfth West And W Howe Park': 1676, 'Ella Bailey Park': 1677, 'Hyperspace': 1678, 'Beacon Alterations': 1679, 'Bayview Playground': 1680, 'Firehouse Mini Park': 1681, 'Seattle Chinese Garden': 1682, "Molbak's Butterfly Garden": 1683, 'Lilac Adult Family Home': 1684, 'Seattle Medical & Rehabilitation Center': 1685, 'Broadway Shoe Repair': 1686, 'Mount Baker Beach': 1687, 'Fremont Peak Park': 1688, 'Arc Watch Works & Engraving': 1689, 'Amasia College': 1690, 'Leschi Lake Dell Natural Area': 1691, 'The Summit At First Hill': 1692, 'Queen Anne': 1693, 'Underground Paranormal Experience': 1694, 'Seaport Elite Yacht Charter': 1695, 'CoppS Hill Terrace': 1696, 'Chelsea Creek': 1697, 'Cambridge College Boston': 1698, 'Forest Hills Station Mall': 1699, 'Cedar Square': 1700, 'Boston Design Center Plaza': 1701, 'Boston College Brighton Campus': 1702, 'Evolving Vox': 1703, 'Samuel Adams Statue': 1704, 'James P Kelleher Rose Garden': 1705, 'Skywalk Observatory': 1706, 'Dreamland Wax Museum': 1707, 'Porzio Park': 1708, 'Rutland Washington Community Garden': 1709, 'Weatherflow Dba Wind Surf': 1710, 'Orient Heights Beach': 1711, 'Boston Museum': 1712, 'Benjamin Franklin Institute Of Technology': 1713, 'Museum of African American History': 1714, 'John Hancock Tower': 1715, 'Savin Maywood Street Garden': 1716, 'Long Lane Meeting House': 1717, 'Massachusetts Historical Society': 1718, 'Nichols House Museum': 1719, "Hillel Foundation of Greater Boston B'nai B'rith": 1720, 'Leo M Birmingham Parkway': 1721, 'Castle Square Parks': 1722, 'Thompson Island': 1723, 'Federal Reserve Plaza': 1724, 'Kennedy Library Harborwalk': 1725, 'Poplar Street Play Area': 1726, 'BU Beach': 1727, 'Fenway Victory Gardens': 1728, 'Pagoda Park': 1729, 'Comm of Mass Dcr': 1730, 'Otis House': 1731, 'Massachusetts Avenue Malls': 1732, 'Arborway Overpass Path': 1733, 'Long Wharf Boat Access': 1734, 'Granary Burying Ground': 1735, 'Fort Warren': 1736, 'West Roxbury H S Campus': 1737, 'Dewey Square Parks': 1738, 'Belle Isle Marsh Reservation': 1739, 'Harvard Stadium': 1740, 'Jamaica Pond Park': 1741, 'Laviscount Park': 1742, 'Marine Park': 1743, 'Bunker Hill Cc Campus Grounds': 1744, 'Commonwealth Museum': 1745, 'Brighton Common': 1746, 'Conley and Tenean Streets Park': 1747, 'Frieda Garcia Park': 1748, 'Msgr Reynolds Playground': 1749, 'Minton Stable Garden': 1750, 'Paul Revere Park': 1751, 'Nova Southeastern University': 1752, 'American College of Greece': 1753, 'Old Harbor Park': 1754, 'Museum Of Afro American History': 1755, 'University Of Massachusetts Boston': 1756, "Lower Allston's Christian Herter Park": 1757, 'Podium Plaza': 1758, 'Bostonian Society Museum Shop': 1759, 'Winthrop Square': 1760, 'Discover the Dinosaurs': 1761, 'Alford Street Bridge': 1762, 'Emerson College': 1763, 'Adams Park': 1764, 'Quincy Street Play Area': 1765, 'Medal Of Honor Park': 1766, 'John F Kennedy Presidential Library And Museum': 1767, 'Thea and James M Stoneman Centennial Park': 1768, 'Jeep Jones Park': 1769, 'Lewis Mall': 1770, 'Copley Square Park': 1771, 'Spangler Food Court at Harvard Business School': 1772, 'Neponset Valley Parkway': 1773, 'Agganis Arena': 1774, 'Kenmore Square': 1775, 'Parker Terrace': 1776, 'Garden of Peace': 1777, 'PattenS Cove': 1778, 'City Hall Plaza': 1779, 'Tent City Courtyards': 1780, 'Shirley Eustis House': 1781, 'Wood Island Bay Marsh': 1782, 'Warren Anatomical Museum': 1783, 'Architectural Heritage Foundation': 1784, 'Puddingstone Park': 1785, 'West End Museum': 1786, 'Msgr Francis A Ryan Park': 1787, 'Eastern National': 1788, 'Umass Boston': 1789, 'Stony Brook Reservation Iii': 1790, 'Historic New England': 1791, 'TD Garden': 1792, 'Paul Revere Mall': 1793, 'Harambee Park': 1794, 'Simmons College Residence Campus': 1795, 'Sullivan Square': 1796, 'Railroad Avenue': 1797, 'Hull Lifesaving Museum': 1798, 'Christopher Columbus Park': 1799, 'Granada Park Garden': 1800, 'Warren Gardens Community Garden': 1801, 'Symphony Road Garden': 1802, 'Franklin Park': 1803, 'Ramsay Park': 1804, 'Pine Street Park': 1805, 'Euston Path Rock': 1806, 'Little Mystic Access Area': 1807, 'Rainsford Island': 1808, 'The Paul Revere House': 1809, 'Nonquit Green': 1810, 'Norman B Leventhal Park': 1811, 'Ringgold Park': 1812, 'Mcphs University': 1813, 'Boston University Sargent Choice Nutrition Center': 1814, 'Harriet Tubman Square': 1815, 'Union Square Plaza': 1816, 'Red Auerbach Concourse': 1817, 'Victory Road Park': 1818, 'Museum Of Fine Arts': 1819, 'Fort Point Associates': 1820, 'Franklin Hill Bha Court': 1821, 'Suffolk University': 1822, 'MassArt Art Museum': 1823, 'Carson Beach': 1824, 'Hobart Park': 1825, 'Leyland Street Garden': 1826, 'Meetinghouse Hill Overlook': 1827, 'OReilly Way Court': 1828, 'South Street Courts': 1829, 'The Sports Museum': 1830, 'Philbrick Schoolyard': 1831, 'Ryan Playground': 1832, 'King School Park': 1833, 'Bright Landry Hockey Center': 1834, 'Lopresti Park': 1835, 'Franklin Park Zoo': 1836, 'Harvard University Harvard Medical School': 1837, 'Boston Long Wharf': 1838, 'Blake Estates Urban Wild': 1839, 'Edgar Allan Poe Statue': 1840, 'Dacia Woodcliff Community Garden': 1841, 'Association of Independent Colleges & Universities of Mass': 1842, 'Charlestown Navy Yard': 1843, 'Public Garden': 1844, 'Massport Harborwalk': 1845, 'Old North Church Museum': 1846, 'Woodhaven': 1847, 'ChildrenS Park': 1848, 'Boston Marathon Attack memorial': 1849, 'African Meeting House': 1850, 'West Street': 1851, 'Mary Soo Hoo Park': 1852, 'Mattahunt School Entrance Plaza': 1853, 'Newbury Street': 1854, 'Cronin Playground': 1855, 'Walden Street Community Garden': 1856, 'Southwest Corridor Community Farm': 1857, 'Mcgann Park': 1858, 'Mattahunt Woods Iii': 1859, 'Core Social Justice Cannabis Museum': 1860, 'Agassiz Road': 1861, 'Rev Loesch Family Park': 1862, 'Anatolia College': 1863, 'Emancipation Statue': 1864, 'The Humps': 1865, 'South Boston Bark Park': 1866, 'SoldierS Monument': 1867, 'Gibson House Museum': 1868, 'Olmsted Green': 1869, 'Bellevue Hill Reservation': 1870, 'Maple Sonoma Streets Community Park': 1871, 'Tortoise and the Hare Sculpture': 1872, 'Forbes Street Garden': 1873, 'Blue Hill Rock': 1874, 'Mcconnell Park': 1875, 'Stanley Ringer Playground': 1876, 'Carleton Court Dog Park': 1877, 'Historic Bostonian Properties': 1878, 'Boston Harbor Walk': 1879, 'Fenway Park': 1880, 'Moynihan Spray Deck': 1881, 'Higginson Schoolyard': 1882, 'Hernandez Schoolyard': 1883, 'Rockledge Street Urban Wild': 1884, 'Boston China Town Neighborhood Center': 1885, 'Malibu Beach': 1886, 'East Boston Greenway': 1887, 'Mattahunt Schoolyard': 1888, 'My Paper Pros': 1889, 'Penniman Road Play Area': 1890, 'Ronan Park': 1891, 'Titus Sparrow Park': 1892, 'Boston History Center And Museum': 1893, 'Southwest Corridor Park': 1894, 'Constitution Beach': 1895, 'Leonard P Zakim Bunker Hill Memorial Bridge': 1896, 'Prince Street Park': 1897, 'Society For the Preservation of New England Antqts': 1898, 'Massachusetts College of Art and Design': 1899, 'Citgo Sign': 1900, 'James Hayes Park': 1901, 'Franklin Hill Green': 1902, 'Paris Fashion Institute': 1903, 'Rink Grounds': 1904, 'John Harvard Mall': 1905, 'Boston University': 1906, 'Forest Hills Preserve': 1907, 'Boston University Plastic & Reconstructive Surgery': 1908, 'Jamaica Plain': 1909, 'Mt Pleasant Play Area': 1910, 'Bremen Street Dog Park': 1911, 'Suffolk University Law School': 1912, 'Boston Business School': 1913, 'Kelly Outdoor Skating Rink': 1914, 'Mass Historical Soc': 1915, 'The Charles River': 1916, 'Boston University School of Theology': 1917, 'Sawmill Brook Brook Farm': 1918, 'Buena Vista': 1919, 'Weider Park': 1920, 'Boyden Park': 1921, 'Baker Chocolate Factory': 1922, 'Mgh Institute Of Health Professions': 1923, 'Cardinal Cushing Memorial Park': 1924, 'Rowe Street Woods': 1925, 'Nira Rock': 1926, 'Boston University School of Public Health': 1927, "The Vilna Shul Boston's Center for Jewish Culture": 1928, 'Commonwealth Pier': 1929, 'Babson Cookson Tract': 1930, 'Russell Fire Club': 1931, 'Belle Isle Coastal Preserve': 1932, 'Old City Hall': 1933, 'Seaport Common': 1934, 'Lavietes Pavilion': 1935, 'Fan Pier Park': 1936, 'Geneva Cliffs': 1937, 'Childe Hassam Park': 1938, 'Good Will Hunting Bench': 1939, "Boston University Women's Council": 1940, 'USS Cassin Young': 1941, 'The Nature Conservancy': 1942, 'Boston Fire Museum': 1943, 'American Legion Highway': 1944, "Jack O' Lantern Journey": 1945, 'Channel Center Dog Park': 1946, 'Case Gym': 1947, 'The Big Dig': 1948, 'Seaport': 1949, 'Camp Meigs': 1950, 'Boston University School Of Medicine': 1951, 'Commercial Point': 1952, 'National Consumer Law Center Inc': 1953, 'Chinese Historical Society of New England': 1954, 'Georges Island': 1955, 'Ryan Play Area': 1956, 'Hayes Park': 1957, 'Edna V Bynoe Park': 1958, 'Clarendon Street Totlot': 1959, 'Carl Yastrzemski Statue': 1960, 'Kendall and Lenox Streets Garden': 1961, 'Bayswater Street': 1962, 'Elc Playlot': 1963, 'Simmons College': 1964, 'Copps Hill Burying Ground': 1965, 'Design Museum Boston': 1966, 'Brighton Hs Hillside': 1967, 'John Copley Statue': 1968, 'Coppens Square': 1969, 'The Esplanade': 1970, 'Reservation Road Park': 1971, 'Massachusetts Democratic Party': 1972, 'Key To Amaze': 1973, 'Admiral Farragut': 1974, 'Faneuil Square': 1975, 'Fire Alarm House Grounds': 1976, 'Lewis Mall Harborpark': 1977, "Boston Children's Museum": 1978, 'Ez Pass': 1979, 'Rogers Park': 1980, 'South Street Community Garden': 1981, 'Langone Park': 1982, 'Stony Brook Commons Park': 1983, 'Sargent Street Park': 1984, 'Mission Main Playlots': 1985, 'Mt Bowdoin Green': 1986, 'Dewey Square Plaza': 1987, 'City Square Park': 1988, 'Downer Dog Park': 1989, 'American Legion Playground': 1990, 'Mansfield Street Dog Park': 1991, 'Reggie Wong Memorial Park': 1992, 'Boston African American National Historic Site': 1993, 'Leland Street Herb Garden': 1994, 'Codman Square': 1995, 'M Street Beach': 1996, 'Mattahunt School Woods': 1997, 'Savin Hill Cove': 1998, 'Joe Moakley Park': 1999, 'Larry Bird Plaque': 2000, 'Fourth Street Park': 2001, 'Boston Tea Party Ship And Museum': 2002, 'Marsh Plaza': 2003, 'Ashburton Place Plaza': 2004, 'Uss Constitution Museum': 2005, 'Kings Chapel Burying Ground': 2006, 'Concourse Ticket Agency': 2007, 'Berklee College Of Music': 2008, 'Herter Park': 2009, 'Old North Memorial Garden': 2010, 'Huntington Hemenway Mall': 2011, 'Symphony Community Park': 2012, 'Williams Street Iii': 2013, 'North End Park': 2014, 'Campus Convenience': 2015, 'Tufts University Boston Campus': 2016, 'Columbia Road Day Boulevard': 2017, 'Stanley Bellevue Park': 2018, 'Copley Place Plaza': 2019, 'Dennis Street Park': 2020, 'Thomas ONeill Tunnel': 2021, 'Hennigan Schoolyard': 2022, 'Constitution Plaza': 2023, 'Commonwealth Avenue Outbound': 2024, 'Stand in Opposition Plaque': 2025, 'Vietnam War Memorial': 2026, 'Smith Park Pumptrack': 2027, 'Flaherty Park': 2028, 'Savin Hill Beach': 2029, 'Rolling Bridge Park': 2030, 'Rose Fitzgerald Kennedy Greenway': 2031, 'Boston University Hillel': 2032, 'Dorchester Heights Monument': 2033, 'Bay State Road': 2034, 'Boston University Academy': 2035, 'School Of The Museum Of Fine Arts At Tufts University': 2036, 'Bill Russell Statue': 2037, 'USS Constitution': 2038, 'Brighton Police Station Campus': 2039, 'Horatio Harris Park': 2040, 'Dorchester Park': 2041, 'Creative Recreational Systems Bridgewater': 2042, 'Oak Square': 2043, 'Lombardi Memorial Park': 2044, 'Boston Nature Center Visitor Ctr': 2045, 'Keycorp': 2046, 'Kidsport': 2047, 'Armenian Heritage Park': 2048, 'Hungarian Monument': 2049, 'Brewer Burroughs Tot Lot': 2050, 'Veterans Memorial Park': 2051, 'Massachusetts State House': 2052, 'William Ellery Channing Monument': 2053, 'Dunbarton Woods': 2054, 'Woodcliff': 2055, 'Forseti Protection Group': 2056, 'Edward Everett Square': 2057, 'Hillside Calumet': 2058, 'Boston Architectural College': 2059, 'Quincy Coleman Garden': 2060, 'Hornblower Yachts': 2061, 'The Boston Common Frog Pond': 2062, 'Brophy Park': 2063, 'Boston Nature Center': 2064, 'Mother Brook Iii': 2065, 'Boston University Bridge': 2066, 'Stonehill Park': 2067, 'Massachusetts Korean War Veterans Memorial': 2068, 'New England Aquarium Whale Watch': 2069, 'Musee Fogg': 2070, 'Umana Schoolyard': 2071, 'Strandway Castle Island': 2072, 'Revere Plaza': 2073, 'I 90 Interchange': 2074, 'Atlantic Avenue Plantings': 2075, 'Ramler Park': 2076, 'United States Government': 2077, 'Edward Everett Hale Statue': 2078, 'Compu Clases': 2079, 'Chinatown Park': 2080, 'Dudley Town Common': 2081, 'Sterling Square': 2082, 'Lo Presti Park': 2083, 'Aquarium New England': 2084, 'Fort Independence Park': 2085, 'Wheelock College of Education & Human Development': 2086, 'Pier 10 Mall': 2087, 'City Square': 2088, 'Isabella Stewart Gardner Museum': 2089, 'Mass Art Campus': 2090, 'Massachusetts Division of Unemployment Assistance': 2091, 'Beethoven School Play Area': 2092, 'Chelsea Street Bridge': 2093, 'Aquatics Unlimited Instructional Products': 2094, 'Joslin Park': 2095, 'Allen Park': 2096, 'Forest Hills Greenspace': 2097, 'Bnan Parcel': 2098, 'Elliot Norton Park': 2099, 'Cutillo Park': 2100, 'Willowwood Rock': 2101, 'Umass Harborwalk': 2102, 'Conley School Play Yard': 2103, 'Southwest Boston Garden Club': 2104, 'Vfw Parkway': 2105, 'Ether Dome': 2106, 'Somerset Street Plaza': 2107, 'Lovells Island': 2108, 'Martini Shell Park Neponset River Reservation': 2109, 'TeenLife Media': 2110, 'Pro Arts Consortium': 2111, 'Havey Beach': 2112, 'Boston University Club': 2113, 'Better Today Addiction Rehab': 2114, 'Woodland Park': 2115, 'Odyssey Sober Living': 2116, 'Northpoint the Evergreen Northgate': 2117, 'South Boston Maritime Park': 2118}
  l_reverse_mapping = {v: k for k, v in l_mapping.items()}
  
  s_mapping={'AM':0,'PM':1}
  sReverseMapping = {v: k for k, v in s_mapping.items()}
  sLabels = [sReverseMapping[i] for i in sorted(sReverseMapping.keys())]
     
  def get_menuType():
      MENU_TYPE = st.selectbox('Menu type 📝', mt_mapping)
      return MENU_TYPE
  def get_dayOfWeek2():
      dayOfWeek = st.selectbox('Day of week 📆', dowLabels,key='tab2_dayOfWeekSelect')
      return dayOfWeek
  def get_TruckBrandName():
      TRUCK_BRAND_NAME = st.selectbox('Truck brand name 🚐', tbn_mapping)
      return TRUCK_BRAND_NAME  
  def get_City():
      CITY = st.selectbox('City 🏙', c_mapping)
      return CITY
  def get_Location(df, CITY):
      locations_for_city = df[df['CITY'] == c_mapping[CITY]]['LOCATION'].unique()
      locations_mapped = [l_reverse_mapping[loc] for loc in locations_for_city]
      LOCATION = st.selectbox('Location 📍', locations_mapped)
      return LOCATION 
  def get_Shift():
      SHIFT = st.selectbox('Shift ⛅️', sLabels)
      return SHIFT
        
  # Define the user input fields
  mt_input = get_menuType()    
  dow_input = get_dayOfWeek2()  
  tbn_input = get_TruckBrandName()  
  c_input = get_City()
  l_input = get_Location(df, c_input)  
  s_input = get_Shift()
    
  # Map user inputs to integer encoding
  mt_int = mt_mapping[mt_input]
  dow_int = dowMapping[dow_input]
  tbn_int = tbn_mapping[tbn_input]
  c_int = c_mapping[c_input]  
  l_int = l_mapping[l_input]    
  s_int = s_mapping[s_input]
  
  prediction_mapping = {0: 'low', 1: 'average', 2: 'high'}
  input_data = [[dow_int,mt_int,tbn_int, c_int,l_int,s_int]]
  input_df = pd.DataFrame(input_data, columns=['DAY_OF_WEEK', 'MENU_TYPE', 'TRUCK_BRAND_NAME', 'CITY', 'LOCATION','SHIFT'])
  
  if st.button('Predict Profits') :
    
      prediction = profit_xinle.predict(input_df)   
      output_data = [mt_int,tbn_int,dow_int, c_int,l_int, s_int, prediction[0]]
      output_df = pd.DataFrame([output_data], columns=['DAY_OF_WEEK', 'MENU_TYPE', 'TRUCK_BRAND_NAME', 'CITY', 'LOCATION','SHIFT','PREDICTED_PROFIT'])
    
      predicted_profit = output_df['PREDICTED_PROFIT'].iloc[0]
      st.write('The predicted profit is {:.2f}.'.format(predicted_profit))

  st.divider()
  st.header("Are customers likely to purchase? 🤷‍♀️") 
  st.write("Profitability alone, whether high or low, does not necessarily indicate high sales. The likelihood of customers purchasing a product is crucial for driving actual sales.")
  if st.button('Predict') :   
      prediction2 = category_xinle.predict(input_df)   
      output_data = [mt_int,tbn_int,dow_int, c_int,l_int,s_int, prediction2[0]]
      output_df2 = pd.DataFrame([output_data], columns=['DAY_OF_WEEK', 'MENU_TYPE', 'TRUCK_BRAND_NAME', 'CITY','LOCATION','SHIFT','PREDICTED_QUANTITY'])
      predicted_quantity = output_df2['PREDICTED_QUANTITY'].map(prediction_mapping).iloc[0]

      # Display the likelihood text with feedback
      if predicted_quantity == 'low':
          likelihood_text = f"The likelihood of customers purchasing is **{predicted_quantity}**. That's bad!"
      elif predicted_quantity == 'average':
          likelihood_text = f"The likelihood of customers purchasing is **{predicted_quantity}**. That's ok!"
      else:
          likelihood_text = f"The likelihood of customers purchasing is **{predicted_quantity}**. That's good!"
      st.markdown(likelihood_text)
      st.write('Here are some recommendation !')


  if st.button('View Recommendations'):
  
      # Initialize lists to store predictions for all combinations
      predicted_profits = []
      predicted_quantities = []
      df_filtered_rows = []
         
      # Loop through every combination of day of the week, truck brand name, city, and shift
      for day_of_week in dowMapping.values():
          for truck_brand_name in tbn_mapping.values():
              for city in c_mapping.values():
                  for shift in s_mapping.values():
                      input_data = [[day_of_week, mt_int, truck_brand_name, city,l_int, shift]]
                      input_df = pd.DataFrame(input_data, columns=['DAY_OF_WEEK', 'MENU_TYPE', 'TRUCK_BRAND_NAME', 'CITY','LOCATION','SHIFT'])
        
                      # Make the predictions
                      prediction1 = profit_xinle.predict(input_df)
                      prediction2 = category_xinle.predict(input_df)
                      predicted_profits.append(prediction1[0])
                      predicted_quantities.append(prediction_mapping[prediction2[0]])
        
                      # Map integer values back to their string representations
                      dow_str = dowReverseMapping[day_of_week]
                      mt_str = list(mt_mapping.keys())[list(mt_mapping.values()).index(mt_int)]
                      tbn_str = list(tbn_mapping.keys())[list(tbn_mapping.values()).index(truck_brand_name)]
                      c_str = list(c_mapping.keys())[list(c_mapping.values()).index(city)]
                      l_str = list(l_mapping.keys())[list(l_mapping.values()).index(l_int)]                   
                      s_str = sReverseMapping[shift]
        
                      df_filtered_rows.append([dow_str, mt_str, tbn_str, c_str,l_str, s_str, prediction1[0], prediction_mapping[prediction2[0]]])
  
          
      # Create a new DataFrame with all combinations
      df_filtered = pd.DataFrame(df_filtered_rows, columns=['DAY_OF_WEEK', 'MENU_TYPE', 'TRUCK_BRAND_NAME', 'CITY','LOCATION', 'SHIFT', 'PREDICTED_PROFIT', 'PREDICTED_QUANTITY'])
  
      # Filter the DataFrame to include only rows where the predicted quantity is 'high'
      high_quantity_rows = df_filtered[df_filtered['PREDICTED_QUANTITY'] == 'high']
          
      # Select the row with the highest predicted profit among the 'high' quantity rows
      row_with_highest_profit = high_quantity_rows.loc[high_quantity_rows['PREDICTED_PROFIT'].idxmax()]
          
      # Extract the required information from the selected row
      day_of_week_recommendation = row_with_highest_profit['DAY_OF_WEEK']
      city_recommendation = row_with_highest_profit['CITY']
      truck_brand_name_recommendation = row_with_highest_profit['TRUCK_BRAND_NAME']
      shift_recommendation = row_with_highest_profit['SHIFT']
      menu_type_recommendation = row_with_highest_profit['MENU_TYPE'] 
          
      # Print the recommendation
      st.markdown(f"Based on the predictions, it is recommended to sell the **{menu_type_recommendation}** menu type in **{city_recommendation}** on\
      **{day_of_week_recommendation}** during the **{shift_recommendation}** shift with the **{truck_brand_name_recommendation}** truck brand. \
      This combination is expected to yield the highest profit and a higher chance of customers making purchases.")


with tab3:
  #from sklift.models import ClassTransformation
  
  #kiara thinks this is painful
  # title
  st.title("Who to upsell and cross-sell to?")
  # description
  st.markdown("""This tab uses insights from conducting ***market baskset analysis*** and feeds them into a uplift machine learning model.
  The model then predicts a customer's propensity score in reaction to upselling and cross-selling promtion.
  For salespeople and marketers, this tab would then categorise these customers into 4 categories:
  1. Persuadables (Customers who buy when promoted to)
  2. Sure things (Customers who buy regardless of promotion)
  3. Lost Causes (Customers who don't buy regardless of promotion)
  4. Sleeping dogs (Customers who don't buy when promoted to)
  """)
  
  # getting the mapping dictionaries from pickle file
  with open("model/enc_dict.pickle","rb") as f:
    unpack = pickle.load(f)
    gender_dict = unpack["gender"]
    marital_dict = unpack["marital"]
    child_dict = unpack["child"]
    menu_dict = unpack["menu"]

  menu_names = {v: k for k, v in menu_dict.items()}

  # retrieving model from pickle files
  arm = pd.read_csv("mba_kiara.csv")
  
  with open("model/uplift_kiara.pickle","rb") as f:
    #slearner = pickle.load(f)
    pass

  # defining user inputs
  def get_gender():
    gender = st.selectbox("Select Gender:", gender_dict, key =1)
    return gender
    
  def get_maritalStatus():
    marital = st.selectbox("Select their marital status:", marital_dict, key =2 )
    return marital
    
  def get_childCount():
    child = st.selectbox("How many children do they have?", child_dict, key = 3)
    return child
    
  def get_Age():
    age = st.number_input(label = "How old are they?", min_value=0, key =4)
    return age
    
  def get_Membership():
    member = st.number_input(label= "How long have they been members?",min_value = 0 ,help="in months", key = 5)
    return member

  c_gender = gender_dict[get_gender()]
  c_marital = marital_dict[get_maritalStatus()]
  c_child = child_dict[get_childCount()]
  c_age = get_Age()
  c_member = get_Membership()

  # get menu item
  first_item = st.selectbox("1st Item in basket?", menu_dict,key = 0)
  second_item = st.selectbox("2nd Item in basket?", menu_dict, key = 9)
  
  if st.button("Suggest!", key= 999):
    # suggest based on market basket analysis
    
    first2 = str(menu_dict[first_item]) + "," + str(menu_dict[second_item])
    
    filtered_df = arm[arm["first2"] == first2].sort_values(by="confidence", ascending = False)

    confidence = 0
    
    # top suggestion 
    top_c = filtered_df.head(1)
    st.write(top_c)
    if len(top_c) > 0:
      consequent = top_c.head(1)["consequents"].item()
      name = menu_names[consequent]
      st.write("Based on market basket analysis, {0} is recommended.".format(name))
      confidence = top_c.head(1)["confidence"].item()
      st.session_state.conf = confidence
      st.session_state.cons =  consequent
    else:
      #consequent = 999
      st.session_state.cons = 999
      st.write("Based on market basket analysis, nothing is suitable as recommendation.")
    
  #consequent = st.selectbox("Item in basket?", menu_dict, key = 8)
  st.write("Should we promote to this customer?")
  #st.write(confidence)
  '''
  #import lightgbm
  
  # predicting
  if st.button("Predict"):
    st.write("TODO: prediction")

    if 'conf' not in st.session_state:
      st.session_state.conf = 0
      
    data_input = [[st.session_state.cons, st.session_state.conf, menu_dict[first_item], second_item, c_gender, c_marital, c_child, c_age, c_member]]
    input_df = pd.DataFrame(data_input, columns = ["consequents","confidence",0,1,"GENDER","MARITAL_STATUS","CHILDREN_COUNT","AGE","MEMBERSHIP"])
    st.write(input_df)
    
    uplift_score = slearner.predict(data_input)
    st.write(uplift_score)
  '''

with tab4:
  import streamlit as st
  import pandas as pd
  import pickle
  import numpy as np
  from sklearn import tree
  import dice_ml
  import IPython
  import math
  
  def main():
      html_temp = """
      <div style="background:#025246 ;padding:10px">
      <h2 style="color:white;text-align:center;"> Tasty Bytes Customer Propencity Prediction </h2>
      </div>
      """
      
      html_div = """
      <div style="background:#35756b ;padding:5px">
      </div>
      """
      
      st.markdown(html_temp, unsafe_allow_html = True)
      st.sidebar.header("Tell us more about yourself")
  
      #input elements
      city_dict = {'Boston':0, 'Denver':0, 'New York City':0, 'San Mateo':0, 'Seattle':0}
      location = st.sidebar.selectbox("Select your city",
          ('Others', 'Boston', 'Denver', 'New York City', 'San Mateo', 'Seattle')
      )
  
      for x in city_dict:
          if(location == x): city_dict.update({x:1});
      #st.write(city_dict) --checking
  
      avg_qty = st.sidebar.slider('How many items do you purchase in a typical transaction?', 0, 20, 5)
      avg_amt = st.sidebar.number_input('How much do you typically spend in a transaction ($)'
                                , min_value = 0.00, step = 0.01, value = 25.00, format = "%.2f")
      
      age = 50 #mean
      gender = 0 #mode (female)
      martial = 0 #mode (single)
      child_count = 0 #mode <but if not single, mode=2>
      freq_cat = 0 #mode (main)
      subcat = 2 #mode (hot)
      
      #input values 
      #Location: OHE (dictionary)
      #Avg amt & Avg qty
      #Age & Gender & Marital & Child count
      #Freq catt & Freq subcat
  
      predict_x = pd.DataFrame({
      "GENDER": gender,
      "MARITAL_STATUS": martial,
      "CHILDREN_COUNT": child_count,
      "AVG_AMT": avg_amt,
      "AVG_QUANTITY": avg_qty,
      "FREQ_CATEGORY": freq_cat,
      "FREQ_SUBCAT": subcat,
      "AGE": age,
      "CITY_New York City": city_dict['New York City'],
      "CITY_Seattle": city_dict['Seattle'],
      "CITY_San Mateo": city_dict['San Mateo'],
      "CITY_Denver": city_dict['Denver'],
      "CITY_Boston": city_dict['Boston']
      }, index = [0])
      
      prediction = predict_model(predict_x);
      #write a message
      if (prediction == 0):
          st.header("You are a LOW spender :man-gesturing-no: :fencer:");
      elif (prediction == 1):
          st.header("You are a HIGH spender :moneybag:");
      else: st.subheader(":red[INVALID prediction output]");
      st.markdown(html_div, unsafe_allow_html = True)
  
      #st.dataframe(predict_x)
      get_counterfactual(predict_x,prediction);
  
  #model deployment
  model = pickle.load(open('./model/CustAnalyV3.2_qiyuan.pkl','rb'))
  
  def predict_model(predict_x):
      #might have to scale values
      #nvm scale is worse
      
      prediction = model.predict(predict_x)
      
      return int(prediction)
  
  def get_counterfactual(predict_x, prediction):
      html_div = """
      <div style="background:#679790 ;padding:5px">
      </div>
      """
      
      ohe_customer_us = pd.read_csv("CustAnaly_newCritV1.1_qiyuan.csv")
      
      d = dice_ml.Data(dataframe=ohe_customer_us.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "CUSTOMER_ID","Recency","MEAN_PROFIT","MEMBER_MONTHS","SPEND_MONTH"]),
                        continuous_features=list(predict_x.columns), outcome_name='SPEND_RANK')
      backend = 'sklearn'
      m = dice_ml.Model(model=model, backend=backend)
      exp = dice_ml.Dice(d,m)
  
      # Generate similar examples
      query_instances = predict_x.iloc[[0]]
      dice_same = exp.generate_counterfactuals(query_instances, total_CFs=5, desired_class=prediction)
      # Visualize counterfactual explanation
      same_df = dice_same.cf_examples_list[0].final_cfs_df
      
      
      # Generate counterfactual examples
      dice_exp = exp.generate_counterfactuals(query_instances, total_CFs=5, desired_class="opposite")
      # Visualize counterfactual explanation
      exp_df = dice_exp.cf_examples_list[0].final_cfs_df
      
  
      #different between user and others
      #calculate % change
      percent_chg_amt = ( predict_x["AVG_AMT"].iloc[0] - exp_df["AVG_AMT"].mean()) / exp_df["AVG_AMT"].mean() * 100
      chg_amt = exp_df["AVG_AMT"].mean() - predict_x["AVG_AMT"].iloc[0]
      percent_chg_qty = math.ceil(exp_df["AVG_QUANTITY"].mean() - predict_x["AVG_QUANTITY"].iloc[0])
  
      st.header('Here\'s a breakdown of your spending habits');
      col1,col2 = st.columns(2)
      if (percent_chg_amt < 0):
          col1.subheader("\nTry spending ${:.2f} more with us :hand_with_index_and_middle_fingers_crossed:".format(chg_amt));
      elif (percent_chg_amt >= 0):
          col1.subheader("\nYou are currently spending ${:.2f} more than others! :muscle:".format(abs(chg_amt)));
      col2.metric("You vs Others", "${:.2f}".format(predict_x["AVG_AMT"].iloc[0]), "{:.2f}%".format(percent_chg_amt));
  
      if (percent_chg_qty == 0):
          st.subheader("\nYour cart size is just right :ok_hand:")
      elif (percent_chg_qty > 0):
          st.subheader("\nYou can put {:d} more items in your basket :shopping_trolley:".format(int(percent_chg_qty)));
      elif (percent_chg_qty<0):
          st.subheader("\nYou have {:d} more items than others! :first_place_medal:".format(abs(int(percent_chg_qty))));
  
      st.markdown(html_div, unsafe_allow_html = True)
      #show examples of other customers
      if (prediction == 0):
          st.subheader("Here are other low spenders... :woman-tipping-hand:");
      elif (prediction == 1):
          st.subheader("Here are other high spenders like you! :money_with_wings:");
      with st.expander ('Guess who!'):
          st.dataframe(same_df.drop(['GENDER','MARITAL_STATUS','CHILDREN_COUNT','AGE'],axis=1))
      
      if (prediction == 0):
          st.subheader("\nHere's how you can be a high spender :gem:");
      elif (prediction == 1):
          st.subheader("\nWatch out! You could become a low spender too! :broken_heart:");
      with st.expander ('Take a peek >_<'):    
          st.dataframe(exp_df.drop(['GENDER','MARITAL_STATUS','CHILDREN_COUNT','AGE'],axis=1))
  
  main();


with tab5:
  #import libs
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
import altair as alt
from streamlit_option_menu import option_menu 
model = pickle.load(open('cust_analysis_RF_Jevan.pkl', 'rb'))
scaler = pickle.load(open('cust_analysis_RF_input.pkl', 'rb'))
#input_data = pd.DataFrame(columns = ['CITY', 'GENDER', 'MARITAL_STATUS', 'CHILDREN_COUNT', 'AVG_AMT', 'AVG_QUANTITY', 'FREQ_CATEGORY', 'FREQ_SUBCAT', 'MEAN_PROFIT', 'DAY_DIFF', 'AGE'])
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Customer Segmentation Model", "Distinct Insights"],
        icons=["house", "person-circle", "pie-chart"]
    )
df = pd.read_csv('rest_customer_us.csv')
high_spender_df = df.loc[df['SPEND_RANK'] == 1]
low_spender_df = df.loc[df['SPEND_RANK'] == 0]
if selected == "Home":
    st.title("Welcome to Jevan's streamlit app!")
    st.sidebar.success("Select a page above.")
    
    st.markdown('<p class="big-font">Is your customer a high spender 💰 or a low spender 👎?</p>', unsafe_allow_html=True)
    st.markdown('<p class="normal-font">This customer segmentation model seeks to divide customers into distinct groups of individuals which in our case is whether a customer is a high or low spender. This will make it easier to target specific groups of customers with tailored products so as to hit our high level goals.</p>', unsafe_allow_html=True)

if selected == "Customer Segmentation Model":
    def manual_standardize(data, means, stds):
        standardized_data = (data - means) / stds
        return standardized_data
    def predict_spend_rank(data):
        # new_input_data_reshaped = data.reshape(1, -1)
        # input_array_scaled = scaler.transform(new_input_data_reshaped)
        means = np.array([9396.284536, 0.643758, 0.899028, 1.093952, 38.506258, 4.148999, 1.868661, 1211.778155, 101.773115, 57.798256, 50.184144])
        stds = np.array([1042.916747, 0.663325, 0.937720, 1.381112, 4.216048, 0.408388, 0.425341, 271.497877, 87.299936, 10.638129, 19.275835])
        standardized_data = manual_standardize(data, means, stds)
        #st.write("Scaled Input Data:")
        #st.write(input_array_scaled)
        prediction = model.predict(standardized_data.reshape(1, -1))
        return int(prediction)
    def main():
            st.title("Customer Segmentation Model")
            st.markdown("""
            <style>
            .big-font {
                font-size:30px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            # Predictor Variables
            freq_subcat_mapping = {
                "Cold Option": 0,
                "Warm Option": 1,
                "Hot Option": 2
            }
            AVG_AMT = 38
            AVG_QUANTITY = 4
            CITY = 10016 #Used the city of San Mateo as it has the highest sales, more room to work on
            DAY_DIFF = st.number_input('What is the average number of days between your first three transactions?', 0, 1000)
            #FREQ_SUBCAT = st.number_input('Input Frequent Subcategory', value = 2)
            FREQ_SUBCAT_options = st.selectbox("What is the frequent subcategory of items you order?", list(freq_subcat_mapping.keys()))
            FREQ_SUBCAT = freq_subcat_mapping[FREQ_SUBCAT_options]            #FREQ_SUBCAT = 2
            MEAN_PROFIT = 7.36
            AGE = 50
            GENDER = 1
            CHILDREN_COUNT = 0
            MARITAL_STATUS = 1
            TOTAL_TRANS = st.number_input('How many transactions have you made with tasty bytes?', 0, 100)
            
            low_spender_html="""
                <div style="background-color:#80ff80; padding:10px >
                <h2 style="color:white;text-align:center;"> The customer is a low spender 👎</h2>
                </div>
            """
            high_spender_html="""
                <div style="background-color:#F4D03F; padding:10px >
                <h2 style="color:white;text-align:center;"> The customer is a high spender 💰</h2>
                </div>
            """
            #Filter df by inputs
            specific_condition_filter = (
                (df['FREQ_SUBCAT'] == FREQ_SUBCAT) &
                (df['DAY_DIFF'] == DAY_DIFF) &
                (df['TOTAL_TRANS'] == TOTAL_TRANS)
            )
            filtered_df = df[specific_condition_filter]
            if(filtered_df.empty):
                fallback_condition1 = (
                    (df['FREQ_SUBCAT'] == FREQ_SUBCAT) &
                    (df['DAY_DIFF'] == DAY_DIFF)
                )
                fallback_condition2 = (
                    (df['FREQ_SUBCAT'] == FREQ_SUBCAT) &
                    (df['TOTAL_TRANS'] == TOTAL_TRANS)
                )
                fallback_condition3 = (
                    (df['DAY_DIFF'] == DAY_DIFF) &
                    (df['TOTAL_TRANS'] == TOTAL_TRANS)
                )
                filtered_df = df[fallback_condition1 | fallback_condition2 | fallback_condition3]
                
            st.subheader("Filtered Data based on User Inputs:")
            st.write(filtered_df)
            if st.button("Predict the spend rank of the customer🏆"):
                input_data = np.asarray([CITY, GENDER, MARITAL_STATUS, CHILDREN_COUNT, AVG_AMT, AVG_QUANTITY, FREQ_SUBCAT, MEAN_PROFIT, DAY_DIFF, TOTAL_TRANS, AGE], dtype = np.float64)
                output = predict_spend_rank(input_data)
                st.success('The spend rank is {}'.format(output))
        
                if output == 0:
                    st.markdown(low_spender_html, unsafe_allow_html=True)
                    text1 = '<p class="normal-font">Customer Behavioural Insights</p>'
                    st.markdown(text1, unsafe_allow_html = True)
                    gender_modes = filtered_df['GENDER'].mode()
                    city_modes = filtered_df['CITY'].mode()
                    marital_status_modes = filtered_df['MARITAL_STATUS'].mode()
                    gender = gender_modes.values[0]
                    city = city_modes.values[0]
                    marital_status = marital_status_modes.values[0]
                    gender_mapping = {
                        0:'Female',
                        1: 'Male',
                        2: 'Undisclosed'
                    }
                    marital_status_mapping = {
                        0 : 'Single',
                        1 : 'Married',
                        2 : 'Divorced/Seperated',
                        3 : 'Undisclosed'
                    }
                    city_mapping = {
                        10613 : 'San Mateo',
                        10016 : 'New York City',
                        9261 : 'Boston',
                        9122 : 'Denver',
                        7288 : 'Seattle'
                    }
                    gender_label = gender_mapping.get(gender)
                    city_label = city_mapping.get(city)
                    marital_status_label = marital_status_mapping.get(marital_status)
                    formatted_insight = "Based on the input, Customer is most likely to be a {} that lives in {} and is {}.".format(gender_label, city_label, marital_status_label) 
                    st.subheader("Customer Behavioural Insights")
                    st.write(formatted_insight)
                elif output == 1:
                    st.markdown(high_spender_html, unsafe_allow_html=True)
                    gender_modes = filtered_df['GENDER'].mode()
                    city_modes = filtered_df['CITY'].mode()
                    marital_status_modes = filtered_df['MARITAL_STATUS'].mode()
                    gender = gender_modes.values[0]
                    city = city_modes.values[0]
                    marital_status = marital_status_modes.values[0]
                    gender_mapping = {
                        0:'Female',
                        1: 'Male',
                        2: 'Undisclosed'
                    }
                    marital_status_mapping = {
                        0 : 'Single',
                        1 : 'Married',
                        2 : 'Divorced/Seperated',
                        3 : 'Undisclosed'
                    }
                    city_mapping = {
                        10613 : 'San Mateo',
                        10016 : 'New York City',
                        9261 : 'Boston',
                        9122 : 'Denver',
                        7288 : 'Seattle'
                    }
                    gender_label = gender_mapping.get(gender)
                    city_label = city_mapping.get(city)
                    marital_status_label = marital_status_mapping.get(marital_status)
                    formatted_insight = "Based on the input, Customer is most likely to be a {} and lives in {} and is generally {}.".format(gender_label, city_label, marital_status_label) 
                    st.subheader("Customer Behavioural Insights")
                    st.write(formatted_insight)
    if __name__ == "__main__":
                main()
if selected == "Distinct Insights":
    st.title("Distinct Insights")
    st.markdown('<p class="big-font">Average Total Transaction by high or low spender</p>', unsafe_allow_html=True)
    spender_avg_trans = df.groupby(['SPEND_RANK'])['TOTAL_TRANS'].mean()
    st.bar_chart(spender_avg_trans)
    text = '<p class="normal-font">Customers who are high spenders💰 typically have an average total number of transactions of <span style="color: green;">63 and above</span> while customers who are low spenders 👎 have an average total number of transactions of <span style="color: red;">51 and below</span>.</p>'
    st.markdown(text, unsafe_allow_html=True)
    day_diff_average = df.groupby(['SPEND_RANK'])['DAY_DIFF'].mean()
    st.markdown('<p class="big-font">Average difference of days in first 3 transactions by high or low spender</p>', unsafe_allow_html=True)
    st.bar_chart(day_diff_average)
    text1 = '<p class="normal-font">Customers who are high spenders💰 typically have an average difference of days in first 3 transactions of <span style="color: green;">76 days and below</span> while customers who are low spenders 👎 have an average total number of transactions of <span style="color: red;">127 days and above</span>.</p>'
    st.markdown(text1, unsafe_allow_html = True)
    subcategory_mapping = {
    0: "Cold Option",
    1: "Warm Option",
    2: "Hot Option"
    }
    # Group data by SPEND_RANK and FREQ_SUBCAT and count occurrences
    spender_freq_subcat_count = df.groupby(['SPEND_RANK', 'FREQ_SUBCAT']).size().reset_index(name='COUNT')
    
    # Update the FREQ_SUBCAT values to their corresponding string labels
    spender_freq_subcat_count['FREQ_SUBCAT_LABEL'] = spender_freq_subcat_count['FREQ_SUBCAT'].map(subcategory_mapping)
    
    # Create a clustered column chart using Altair
    chart = alt.Chart(spender_freq_subcat_count).transform_calculate(
        FREQ_SUBCAT_LABEL='datum.FREQ_SUBCAT_LABEL'
    ).mark_bar().encode(
        x=alt.X('FREQ_SUBCAT_LABEL:N', title='Frequent Subcategory'),
        y=alt.Y('COUNT:Q', title='Count'),
        color=alt.Color('SPEND_RANK:N', scale=alt.Scale(domain=['0', '1'], range=['red', 'green']), legend=alt.Legend(title='Spend Rank'))
    ).properties(
        width=600,
        height=400,
        title="Count of High💰 and Low👎 Spenders Based on Frequent Subcategory"
    )
    
    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

    
    text = '<p class="normal-font">In general, being a high spender💰 or low spender👎 does not affect the choice of subcategory of items they are purchasing. The subcategory of items with the most count of people buying would be the <span style="color: green;">hot option</span>☕.</p>'
    st.markdown(text, unsafe_allow_html=True)
