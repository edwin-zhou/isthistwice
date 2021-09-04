// This file can be replaced during build by using the `fileReplacements` array.
// `ng build` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

export const environment = {
  production: false,
  mainURL: 'http://localhost:3000',
  modelURL: "http://localhost:3000/models/ot9/model.json",


  LABELS: ['Tzuyu', 'Chaeyoung', 'Dahyun', 'Mina', 'Jihyo', 'Sana', 'Momo', 'Jeongyeon', 'Nayeon'],
  colors: ["Blue", "Red", "White", "Mint", "Apricot", "Purple", "Pink", "Yellow-Green", "Sky-Blue"],
  colorsRGB: ["#0000ff", "#ff0000", "#FFFFFF", "#3EB489", "#FBCEB1", "#800080", "#FFC0CB", "#9acd32", "#87CEEB"]
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/plugins/zone-error';  // Included with Angular CLI.
