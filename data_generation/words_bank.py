placements = ['on a table', 'on a shelf', 'on a wooden table', 'on the street', 'on the forest bed', 'on the floor',
              'on the ground', 'on the grass', 'on the sand', 'on the beach', 'on the shore', 'on the pavement',
              'in a room', 'in a living room', 'in a bedroom', 'in a kitchen', 'in a library', 'in a bathroom',
              'in a garden', 'in a park', 'in a library', 'in a bathroom', 'in a garden', 'in a park', 'in the office',
              'in the classroom', 'in the cafe', 'in the restaurant', 'on the balcony', 'on the rooftop',
              'in the hallway', 'on the staircase', 'in the elevator', 'in the lobby', 'in the garage',
              'in the basement', 'in the attic', 'on the porch', 'on the deck', 'on the patio', 'on the mountain',
              'in the valley', 'in the forest', 'in the jungle', 'on the riverbank', 'at the waterfall', 'by the lake',
              'by the pond', 'in the desert', 'in the canyon', 'on the glacier', 'by the iceberg', 'in the field',
              'on the farm', 'in the vineyard', 'in the orchard', 'at the playground', 'in the stadium', 'in the arena',
              'in the gym', 'in the pool', 'in the sauna', 'in the spa', 'in the beauty salon', 'in the bookstore',
              'in the grocery store', 'in the market', 'in the mall', 'in the theater', 'in the cinema',
              'in the studio', 'in the gallery', 'in the museum', 'at the aquarium', 'at the zoo',
              'at the amusement park', 'at the campsite', 'in the RV park', 'at the resort', 'in the hotel',
              'in the motel', 'in the inn', 'in the lodge', 'in the hostel', 'at the airport', 'at the train station',
              'at the bus station', 'at the port', 'in the church', 'in the temple', 'in the mosque',
              'in the synagogue', 'at the university', 'at the school', 'in the daycare', 'in the nursery',
              'in the factory', 'in the warehouse', 'in the office building', 'in the skyscraper', 'in the cottage',
              'in the bungalow', 'in the mansion', 'in the castle']

texture_attributes = ['glossy', 'shiny', 'matte', 'rough', 'bumpy', 'smooth', 'shimmering', 'sparkling', 'dusty',
                      'fuzzy', 'soft', 'hard', 'brittle', 'flexible', 'elastic', 'stiff', 'rigid', 'tough', 'flimsy',
                      'delicate', 'fragile', 'sturdy', 'solid', 'hollow', 'dense', 'light', 'heavy', 'grainy',
                      'slippery', 'sticky', 'warm', 'cool', 'metallic', 'pearlescent', 'satin', 'dull', 'reflective',
                      'translucent', 'opaque', 'homogeneous', 'heterogeneous', 'veined', 'flat', 'raised', 'textured',
                      'rugged', 'water-resistant', 'light-absorbing', 'dust-repellent', 'cozy', 'luxurious', 'rustic',
                      'weathered', 'polished', 'aged', 'green', 'brown', 'yellow', 'orange', 'red', 'blue', 'purple',
                      'pink', 'white', 'black', 'grey', 'silver', 'gold', 'cyan', 'magenta', 'turquoise', 'ivory',
                      'tan', 'beige', 'navy', 'maroon', 'charcoal', 'teal', 'olive', 'peach', 'lavender', 'uniform',
                      'abstract', 'geometric', 'random', 'symmetrical', 'asymmetrical', 'striped', 'checked', 'paisley',
                      'plaid', 'houndstooth', 'polka-dot', 'herringbone', 'chevron', 'argyle', 'non-reflective',
                      'marbled', 'grained', 'streaked', 'spotted', 'speckled', 'dappled', 'mottled', 'flecked',
                      'patched', 'layered', 'deep', 'shallow', 'pitted', 'embossed', 'engraved', 'new', 'old', 'worn',
                      'damaged', 'antique', 'vintage', 'distressed', 'refurbished', 'restored', 'pristine',
                      'immaculate', 'tarnished', 'faded', 'marble', 'granite', 'wood', 'metal', 'glass', 'plastic',
                      'fabric', 'leather', 'paper', 'cardboard', 'concrete', 'brick', 'stone', 'sand', 'dirt', 'mud',
                      'clay', 'ceramic', 'porcelain', 'rubber', 'sponge', 'foam', 'felt', 'velvet', 'silk', 'cotton',
                      'wool', 'linen', 'denim', 'lace', 'tweed', 'nylon', 'polyester', 'acrylic', 'spandex', 'suede',
                      'mesh', 'bamboo', 'hemp', 'leaf pattern', 'floral pattern', 'animal pattern', 'zebra pattern',
                      'tiger pattern', 'leopard pattern', 'cheetah pattern', 'giraffe pattern', 'snake pattern',
                      'crocodile pattern', 'camouflage pattern', 'mosaic pattern', 'kaleidoscope pattern',
                      'mandala pattern', 'tartan pattern', 'batik pattern', 'ikat pattern', 'quilted pattern',
                      'glittering', 'mirrored', 'satin-finish', 'carbon fiber', 'knitted', 'crocheted', 'embroidered',
                      'pleated', 'crinkled', 'crumpled', 'woven', 'braided', 'perforated', 'padded', 'quilted',
                      'thermal', 'insulated', 'gauzy', 'translucent-finish', 'iridescent', 'opalescent', 'neon',
                      'pastel', 'vibrant', 'dull-finish', 'chalky', 'silky-smooth', 'rubbery', 'gummy', 'waxy', 'oily',
                      'soapy', 'milky', 'crystal-clear', 'frosted', 'etched', 'blurred', 'swirled', 'twisted', 'coiled',
                      'looped', 'interwoven', 'knotted', 'spiral', 'diagonal', 'crosshatched', 'lacy', 'beaded',
                      'sequined', 'flocked', 'brushed metal', 'anodized', 'galvanized', 'powder-coated', 'acid-washed',
                      'sun-bleached', 'peeling', 'cracked', 'chipped', 'burnished', 'oxidized', 'corroded', 'stained',
                      'dyed', 'tie-dye pattern', 'ombre', 'gradient', 'speckled paint', 'splattered paint',
                      'marbleized', 'woodgrain', 'cork', 'terrazzo', 'bamboo texture', 'reed', 'sisal', 'sea grass',
                      'jute', 'chalkboard', 'magnetic', 'glazed', 'unglazed', 'raw', 'burnt', 'smoked', 'sanded',
                      'planed', 'rough-cut', 'varnished', 'unvarnished', 'waxed', 'oil-finished', 'shellac-finished',
                      'lacquered', 'patina', 'brushed', 'hammered', 'spun', 'wrought', 'forged', 'cast', 'molded',
                      '3D printed', 'laminated', 'veneered', 'inlaid', 'gilded', 'silvered', 'leafed', 'foiled',
                      'embossed pattern', 'debossed pattern', 'puzzle pattern', 'geometric pattern',
                      'optical illusion pattern', 'holographic pattern', 'psychedelic pattern', 'pop art pattern',
                      'art deco pattern', 'Victorian pattern', 'Baroque pattern', 'Renaissance pattern',
                      'gothic pattern', 'Celtic pattern', 'tribal pattern', 'Ethnic pattern', 'folk pattern',
                      'historical pattern']

adjectives = ['rotten', 'big', 'small', 'many', 'burning', 'melting', 'shattered', 'dried', 'sliced', 'moldy',
              'glistening', 'fluffy', 'plush', 'opaque', 'wrinkled', 'frosted', 'antique', 'futuristic', 'cracked',
              'glowing', 'glossy', 'translucent', 'gothic', 'young', 'old', 'rustic', 'two', 'multiple', 'group',
              'shiny', 'dull', 'colorful', 'floating', 'winged', 'soggy', 'ancient', 'tiny', 'enormous', 'skeletal',
              'hairless', 'furry', 'grimy', 'frozen', 'dusty', 'muddy', 'bubbly', 'spiky', 'slimy', 'scaly', 'feathery',
              'hairy', 'fuzzy', 'smooth', 'rough', 'gleaming', 'heavy', 'wet', 'dry', 'aged', 'transparent', 'empty',
              'full']

art_types = ['photo', 'painting', 'sketch', 'sculpture', 'photograph', 'drawing', 'tapestry', 'mosaic', 'carving',
             'pottery', 'ceramic', 'origami', 'stained glass', 'engraving', 'watercolor painting', 'oil painting',
             'acrylic painting', 'charcoal drawing', 'pencil drawing', 'pastel drawing', 'ink drawing',
             'digital painting', 'collage', 'mixed media', 'woodcut', 'lithograph', 'etching', 'engraving',
             'comic drawing', 'cartoon', 'animation', 'illustration', 'concept art', 'conceptual art', '3D rendering',
             'digital art', 'traditional art', 'abstract art', 'realistic art', 'minimalist art', 'abstract art'
             ]
