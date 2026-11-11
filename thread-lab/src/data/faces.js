// Faces for the balls: 32 photos from Wikimedia Commons, each cropped to the face and credited in the page footer.
const FACES = [
  {
    "name": "Taylor Swift",
    "src": "src/assets/faces/01-taylor-swift.webp",
    "title": "Taylor Swift (52791470537) (cropped).jpg",
    "creator": "Ronald Woan",
    "license": "CC BY-SA 2.0",
    "source": "https://commons.wikimedia.org/wiki/File:Taylor_Swift_(52791470537)_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/2.0/"
  },
  {
    "name": "Beyoncé",
    "src": "src/assets/faces/02-beyonce.webp",
    "title": "Beyoncé - Tottenham Hotspur Stadium 2023 (cropped).jpg",
    "creator": "Raph_PH",
    "license": "CC BY 2.0",
    "source": "https://commons.wikimedia.org/wiki/File:Beyonc%C3%A9_-_Tottenham_Hotspur_Stadium_2023_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/2.0/"
  },
  {
    "name": "Ariana Grande",
    "src": "src/assets/faces/03-ariana-grande.webp",
    "title": "Ariana Grande Eternal Sunshine Tour performing.jpg",
    "creator": "Saturnmondays",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Ariana_Grande_Eternal_Sunshine_Tour_performing.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Sabrina Carpenter",
    "src": "src/assets/faces/04-sabrina-carpenter.webp",
    "title": "Sabrina Carpenter at BST Hyde Park - 20250706 - 033.jpg",
    "creator": "Raph_PH",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Sabrina_Carpenter_at_BST_Hyde_Park_-_20250706_-_033.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Olivia Rodrigo",
    "src": "src/assets/faces/05-olivia-rodrigo.webp",
    "title": "Olivia Rodrigo 3.jpeg",
    "creator": "Raph_PH",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Olivia_Rodrigo_3.jpeg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Billie Eilish",
    "src": "src/assets/faces/06-billie-eilish.webp",
    "title": "BillieEilishO2140725-39 - 54665577407 (cropped).jpg",
    "creator": "Raph_PH",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:BillieEilishO2140725-39_-_54665577407_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Chappell Roan",
    "src": "src/assets/faces/07-chappell-roan.webp",
    "title": "Chappell Roan, 20 August 2025 07.jpg",
    "creator": "Raph_PH",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Chappell_Roan,_20_August_2025_07.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Zendaya",
    "src": "src/assets/faces/08-zendaya.webp",
    "title": "Zendaya-byPhilipRomano.jpg",
    "creator": "PhilipRomano",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Zendaya-byPhilipRomano.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Jenna Ortega",
    "src": "src/assets/faces/09-jenna-ortega.webp",
    "title": "Jenna Ortega at the 2026 Sundance Film Festival (cropped).jpg",
    "creator": "Colleen Sturtevant",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Jenna_Ortega_at_the_2026_Sundance_Film_Festival_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Simone Biles",
    "src": "src/assets/faces/10-simone-biles.webp",
    "title": "Simone Biles Worlds 2026.jpg",
    "creator": "Ocoudis",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Simone_Biles_Worlds_2026.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "SZA",
    "src": "src/assets/faces/11-sza.webp",
    "title": "KendrickSZASPurs230725-19 - 54683179509 (cropped) (cropped).jpg",
    "creator": "Raph_PH",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:KendrickSZASPurs230725-19_-_54683179509_(cropped)_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Bad Bunny",
    "src": "src/assets/faces/12-bad-bunny.webp",
    "title": "Bad Bunny 2019 by Glenn Francis.jpg",
    "creator": "Glenn Francis / Toglenn (www.PacificProDigital.com)",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Bad_Bunny_2019_by_Glenn_Francis.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Kendrick Lamar",
    "src": "src/assets/faces/13-kendrick-lamar.webp",
    "title": "KendrickSZASPurs230725-22 - 54683179414 (portrait crop).jpg",
    "creator": "Raph_PH",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:KendrickSZASPurs230725-22_-_54683179414_(portrait_crop).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "The Weeknd",
    "src": "src/assets/faces/14-the-weeknd.webp",
    "title": "The Weeknd Portrait by Brian Ziff.jpg",
    "creator": "Brian Ziff",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:The_Weeknd_Portrait_by_Brian_Ziff.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Harry Styles",
    "src": "src/assets/faces/15-harry-styles.webp",
    "title": "Harry Styles (2026).jpg",
    "creator": "Wilnel José Verdú Guerrero",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Harry_Styles_(2026).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Bruno Mars",
    "src": "src/assets/faces/16-bruno-mars.webp",
    "title": "Bruno Mars - 24K Magic Tour (37320927992) (cropped).jpg",
    "creator": "slgckgc / Sharon L. Chapman",
    "license": "CC BY 2.0",
    "source": "https://commons.wikimedia.org/wiki/File:Bruno_Mars_-_24K_Magic_Tour_(37320927992)_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/2.0/"
  },
  {
    "name": "Timothée Chalamet",
    "src": "src/assets/faces/17-timothee-chalamet.webp",
    "title": "Timothée Chalamet 2025 2 (cropped).jpg",
    "creator": "Amy Martin Photography",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Timoth%C3%A9e_Chalamet_2025_2_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Tom Holland",
    "src": "src/assets/faces/18-tom-holland.webp",
    "title": "TomHolland-byPhilipRomano3.jpg",
    "creator": "PhilipRomano",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:TomHolland-byPhilipRomano3.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Pedro Pascal",
    "src": "src/assets/faces/19-pedro-pascal.webp",
    "title": "Pedro Pascal at the 2025 Cannes Film Festival 02.jpg",
    "creator": "Gabriel Hutchinson",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Pedro_Pascal_at_the_2025_Cannes_Film_Festival_02.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Michael B Jordan",
    "src": "src/assets/faces/20-michael-b-jordan.webp",
    "title": "Michael B Jordan - Sinners (cropped).jpg",
    "creator": "Kevin Paul",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Michael_B_Jordan_-_Sinners_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "LeBron James",
    "src": "src/assets/faces/21-lebron-james.webp",
    "title": "LeBron James (55095816551).jpg",
    "creator": "Erik Drost",
    "license": "CC BY 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:LeBron_James_(55095816551).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"
  },
  {
    "name": "Stephen Curry",
    "src": "src/assets/faces/22-stephen-curry.webp",
    "title": "Stephen Curry, Olympic Games 2024.jpg",
    "creator": "Clément Bardot",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Stephen_Curry,_Olympic_Games_2024.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Sam Smith",
    "src": "src/assets/faces/23-sam-smith.webp",
    "title": "SamSmith-byPhilipRomano (cropped).jpg",
    "creator": "PhilipRomanoPhoto",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:SamSmith-byPhilipRomano_(cropped).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Demi Lovato",
    "src": "src/assets/faces/24-demi-lovato.webp",
    "title": "Demi Lovato Interview Feb 2020.png",
    "creator": "Ashley Graham",
    "license": "CC BY 3.0",
    "source": "https://commons.wikimedia.org/wiki/File:Demi_Lovato_Interview_Feb_2020.png",
    "licenseUrl": "https://creativecommons.org/licenses/by/3.0/"
  },
  {
    "name": "Janelle Monae",
    "src": "src/assets/faces/25-janelle-monae.webp",
    "title": "Janelle monae 9033260.jpg",
    "creator": "Frypie",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Janelle_monae_9033260.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Bella Ramsey",
    "src": "src/assets/faces/26-bella-ramsey.webp",
    "title": "Bella Ramsey in 2026 01.jpg",
    "creator": "Mateusz Malta",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Bella_Ramsey_in_2026_01.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Emma D'Arcy",
    "src": "src/assets/faces/27-emma-d-arcy.webp",
    "title": "Emma D'Arcy Uk.jpg",
    "creator": "Sophie Vittoria",
    "license": "CC0 1.0 Universal",
    "source": "https://commons.wikimedia.org/wiki/File:Emma_D%27Arcy_Uk.jpg",
    "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/"
  },
  {
    "name": "Emma Corrin",
    "src": "src/assets/faces/28-emma-corrin.webp",
    "title": "Emma Corrin-08687.jpg",
    "creator": "CaseyJNewell",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Emma_Corrin-08687.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  },
  {
    "name": "Amandla Stenberg",
    "src": "src/assets/faces/29-amandla-stenberg.webp",
    "title": "Amandla Stenberg by Gage Skidmore.jpg",
    "creator": "Gage Skidmore",
    "license": "CC BY-SA 3.0",
    "source": "https://commons.wikimedia.org/wiki/File:Amandla_Stenberg_by_Gage_Skidmore.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/3.0/"
  },
  {
    "name": "Jonathan Van Ness",
    "src": "src/assets/faces/30-jonathan-van-ness.webp",
    "title": "Jonathan Van Ness, 2018-04.jpg",
    "creator": "Cntrl+Alt+Delete",
    "license": "CC BY 3.0",
    "source": "https://commons.wikimedia.org/wiki/File:Jonathan_Van_Ness,_2018-04.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by/3.0/"
  },
  {
    "name": "Liv Hewson",
    "src": "src/assets/faces/31-liv-hewson.webp",
    "title": "Liv Hewson (41163729811).jpg",
    "creator": "Greg2600 (https://www.flickr.com/photos/greg2600)",
    "license": "CC BY-SA 2.0",
    "source": "https://commons.wikimedia.org/wiki/File:Liv_Hewson_(41163729811).jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/2.0/"
  },
  {
    "name": "Lachlan Watson",
    "src": "src/assets/faces/32-lachlan-watson.webp",
    "title": "Lachlan Watson.jpg",
    "creator": "MalachiColey",
    "license": "CC BY-SA 4.0",
    "source": "https://commons.wikimedia.org/wiki/File:Lachlan_Watson.jpg",
    "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/"
  }
];
